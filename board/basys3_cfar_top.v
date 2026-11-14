// On-board test of cfar_ca_1d on a Digilent Basys 3 (xc7a35tcpg236-1, 100 MHz).
// A block-RAM ROM (cfar_rom.hex: one entry per cell, bit 16 = start of line, bits 15:0 =
// cell power) is streamed through the CFAR at one cell per clock; every decision
// {o_idx, o_det} is captured in block RAM; a host script triggers runs and reads results
// over the USB-UART (115200 8N1) and compares them with the Python fixed-point model.
// Commands: 'R' run without stalls, 'S' run with LFSR stalls (~25% idle cycles),
//           'D' dump captured decisions (2 bytes each: {3'b0, idx[11:0], det}).
// Run reply: 'K', n_dec[15:0], cycles[31:0] (first input to last decision, inclusive),
//            n_in[15:0], stall_cycles[31:0], in_cycles[31:0] (first to last input, inclusive)
//            (big-endian, 17 bytes).
// LEDs: [0] running, [1] done, [15:2] decisions captured (MSBs).
`timescale 1ns / 1ps
module basys3_cfar_top #(
    parameter integer N_CELLS = 16384,
    parameter integer N_TRAIN = 16,
    parameter integer N_GUARD = 2,
    parameter [16:0]  ALPHA_Q = 17'd43715,     // round(10.6727 * 4096): N = 32, Pfa 1e-4
    parameter integer CLKS_PER_BIT = 868,      // 100 MHz / 115200
    parameter         ROM_FILE = "cfar_rom.hex"
) (
    input  wire        clk,
    input  wire        btnC,
    input  wire        RsRx,
    output wire        RsTx,
    output wire [15:0] led
);
    localparam integer AW = 14;                // 16384-entry ROM / capture RAM

    // ---------------- reset synchronizer
    reg [2:0] rst_sync = 3'b111;
    always @(posedge clk) rst_sync <= {rst_sync[1:0], btnC};
    wire rst = rst_sync[2];

    // ---------------- UART
    wire       rx_valid;
    wire [7:0] rx_data;
    reg        tx_start;
    reg  [7:0] tx_data;
    wire       tx_busy;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (.clk(clk), .rst(rst), .rx(RsRx), .valid(rx_valid), .data(rx_data));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (.clk(clk), .rst(rst), .start(tx_start), .data(tx_data), .tx(RsTx), .busy(tx_busy));

    // ---------------- stimulus ROM (block RAM, registered read)
    (* rom_style = "block" *) reg [16:0] rom [0:N_CELLS-1];
    initial $readmemh(ROM_FILE, rom);
    reg [AW-1:0] rd_addr;
    reg [16:0]   rom_q;
    always @(posedge clk) rom_q <= rom[rd_addr];

    // ---------------- capture RAM
    (* ram_style = "block" *) reg [12:0] cap [0:(1<<AW)-1];
    reg [AW:0]   n_dec;
    reg [AW-1:0] dump_addr;
    reg [12:0]   cap_q;
    wire         o_valid, o_det;
    wire [11:0]  o_idx;
    always @(posedge clk) begin
        if (o_valid && !n_dec[AW]) cap[n_dec[AW-1:0]] <= {o_idx, o_det};
        cap_q <= cap[dump_addr];
    end

    // ---------------- CFAR under test
    reg  cfar_rst;
    reg  issue;                 // a ROM read was issued last cycle -> data valid this cycle
    cfar_ca_1d #(.N_TRAIN(N_TRAIN), .N_GUARD(N_GUARD)) dut (
        .clk(clk), .rst(cfar_rst), .alpha_q(ALPHA_Q), .i_valid(issue), .i_sol(rom_q[16]),
        .i_data(rom_q[15:0]), .o_valid(o_valid), .o_idx(o_idx), .o_det(o_det));

    // ---------------- LFSR for stall pattern (x^16 + x^14 + x^13 + x^11 + 1)
    reg [15:0] lfsr;
    wire stall_now = (lfsr[1:0] == 2'b00);     // ~25 % of cycles idle in stall mode

    // ---------------- control FSM
    localparam S_IDLE = 0, S_RST = 1, S_RUN = 2, S_DRAIN = 3, S_REPLY = 4, S_DUMP = 5;
    reg [2:0]  st;
    reg [3:0]  rcnt;
    reg        stall_mode, first_seen;
    reg [AW:0] n_in;
    reg [31:0] cyc, cyc_first, cyc_last, cyc_in_last, n_stall;
    reg [4:0]  bidx;            // reply byte index
    reg [AW:0] didx;            // dump entry index
    reg        dbyte, dwait;
    reg [7:0]  drain;

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; cfar_rst <= 1'b1; issue <= 1'b0; rd_addr <= 0; n_in <= 0; n_dec <= 0;
            tx_start <= 1'b0; lfsr <= 16'hACE1; cyc <= 0; first_seen <= 1'b0; stall_mode <= 1'b0;
            dump_addr <= 0; n_stall <= 0;
        end else begin
            tx_start <= 1'b0;
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
            cyc <= cyc + 1'b1;
            if (o_valid && !n_dec[AW]) begin
                n_dec <= n_dec + 1'b1;
                cyc_last <= cyc;
            end
            issue <= 1'b0;
            case (st)
                S_IDLE: begin
                    cfar_rst <= 1'b0;
                    if (rx_valid && (rx_data == "R" || rx_data == "S")) begin
                        stall_mode <= (rx_data == "S");
                        st <= S_RST; rcnt <= 0; cfar_rst <= 1'b1;
                        rd_addr <= 0; n_in <= 0; n_dec <= 0; first_seen <= 1'b0; n_stall <= 0;
                    end else if (rx_valid && rx_data == "D") begin
                        st <= S_DUMP; didx <= 0; dump_addr <= 0; dbyte <= 1'b0; dwait <= 1'b1;
                    end
                end
                S_RST: begin
                    rcnt <= rcnt + 1'b1;
                    if (rcnt == 4'd7) begin cfar_rst <= 1'b0; st <= S_RUN; end
                end
                S_RUN: begin
                    if (n_in == N_CELLS) begin
                        st <= S_DRAIN; drain <= 0;
                    end else if (stall_mode && stall_now) begin
                        n_stall <= n_stall + 1'b1;
                    end else begin
                        issue <= 1'b1;                  // data for rd_addr appears next cycle
                        if (!first_seen) begin first_seen <= 1'b1; cyc_first <= cyc + 1'b1; end
                        cyc_in_last <= cyc + 1'b1;
                        rd_addr <= rd_addr + 1'b1;
                        n_in <= n_in + 1'b1;
                    end
                end
                S_DRAIN: begin
                    drain <= drain + 1'b1;
                    if (drain == 8'd31) begin st <= S_REPLY; bidx <= 0; end
                end
                S_REPLY: if (!tx_busy && !tx_start) begin
                    tx_start <= 1'b1;
                    case (bidx)
                        0: tx_data <= "K";
                        1: tx_data <= n_dec >> 8;
                        2: tx_data <= n_dec[7:0];
                        3: tx_data <= (cyc_last - cyc_first + 1) >> 24;
                        4: tx_data <= (cyc_last - cyc_first + 1) >> 16;
                        5: tx_data <= (cyc_last - cyc_first + 1) >> 8;
                        6: tx_data <= (cyc_last - cyc_first + 1);
                        7: tx_data <= n_in >> 8;
                        8: tx_data <= n_in[7:0];
                        9: tx_data <= n_stall >> 24;
                        10: tx_data <= n_stall >> 16;
                        11: tx_data <= n_stall >> 8;
                        12: tx_data <= n_stall[7:0];
                        13: tx_data <= (cyc_in_last - cyc_first + 1) >> 24;
                        14: tx_data <= (cyc_in_last - cyc_first + 1) >> 16;
                        15: tx_data <= (cyc_in_last - cyc_first + 1) >> 8;
                        default: tx_data <= (cyc_in_last - cyc_first + 1);
                    endcase
                    if (bidx == 5'd16) st <= S_IDLE;
                    bidx <= bidx + 1'b1;
                end
                S_DUMP: begin
                    if (didx == n_dec) begin
                        st <= S_IDLE;
                    end else if (dwait) begin
                        dwait <= 1'b0;                  // cap_q valid one cycle after dump_addr
                    end else if (!tx_busy && !tx_start) begin
                        tx_start <= 1'b1;
                        tx_data <= dbyte ? cap_q[7:0] : {3'b000, cap_q[12:8]};
                        if (dbyte) begin
                            didx <= didx + 1'b1;
                            dump_addr <= dump_addr + 1'b1;
                            dwait <= 1'b1;
                        end
                        dbyte <= ~dbyte;
                    end
                end
                default: st <= S_IDLE;
            endcase
        end
    end

    assign led = {n_dec[AW:AW-13], (st == S_IDLE) && (n_dec != 0), (st == S_RUN) || (st == S_DRAIN)};
endmodule
