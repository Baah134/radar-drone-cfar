// File-driven testbench for cfar_ca_1d (Vivado xsim or any SV simulator).
// Input file (VEC_IN): first line "ALPHA <hex>", then one cell per line: "<sol> <hex data>".
// Output file (VEC_OUT): one line per decision: "<idx> <det>".
// Random idle cycles (i_valid = 0) are inserted with probability STALL_PCT/100 to check
// that stalls do not change results. Assertions: running sums never exceed their bound.
`timescale 1ns / 1ps
module tb_cfar_ca_1d;
    parameter integer DATA_W    = 16;
    parameter integer N_TRAIN   = 16;
    parameter integer N_GUARD   = 2;
    parameter integer ALPHA_W   = 17;
    parameter integer STALL_PCT = 20;
    parameter integer SEED      = 1;
    parameter VEC_IN  = "vec_in.txt";
    parameter VEC_OUT = "vec_out.txt";

    localparam integer SIDE_W   = DATA_W + $clog2(N_TRAIN + 1);
    localparam longint SIDE_MAX = longint'(N_TRAIN) * ((longint'(1) << DATA_W) - 1);

    reg clk = 0, rst = 1;
    reg [ALPHA_W-1:0] alpha_q = 0;
    reg i_valid = 0, i_sol = 0;
    reg [DATA_W-1:0] i_data = 0;
    wire o_valid, o_det;
    wire [11:0] o_idx;

    cfar_ca_1d #(.DATA_W(DATA_W), .N_TRAIN(N_TRAIN), .N_GUARD(N_GUARD), .ALPHA_W(ALPHA_W)) dut (
        .clk(clk), .rst(rst), .alpha_q(alpha_q), .i_valid(i_valid), .i_sol(i_sol), .i_data(i_data),
        .o_valid(o_valid), .o_idx(o_idx), .o_det(o_det));

    always #5 clk = ~clk;

    integer fin, fout, r, sol, dat, n_in = 0, n_out = 0, errors = 0, seed;
    reg [8*16-1:0] tag;

    // overflow / invariant checks on the running sums
    always @(posedge clk) if (!rst) begin
        if (longint'(dut.lead_sum) > SIDE_MAX || longint'(dut.lag_sum) > SIDE_MAX) begin
            $display("ERROR: running sum exceeds bound: lead=%0d lag=%0d max=%0d", dut.lead_sum, dut.lag_sum, SIDE_MAX);
            errors = errors + 1;
        end
    end

    always @(posedge clk) if (o_valid) begin
        $fdisplay(fout, "%0d %0d", o_idx, o_det);
        n_out = n_out + 1;
    end

    initial begin
        seed = SEED;
        fin = $fopen(VEC_IN, "r");
        fout = $fopen(VEC_OUT, "w");
        if (fin == 0 || fout == 0) begin $display("ERROR: cannot open vector files"); $finish; end
        r = $fscanf(fin, "%s %h\n", tag, alpha_q);
        repeat (3) @(posedge clk);
        rst <= 0;
        @(posedge clk);
        while (!$feof(fin)) begin
            r = $fscanf(fin, "%d %h\n", sol, dat);
            if (r != 2) break;
            while (($urandom(seed) % 100) < STALL_PCT) begin
                i_valid <= 0; @(posedge clk); seed = seed + 1;
            end
            i_valid <= 1; i_sol <= sol[0]; i_data <= dat[DATA_W-1:0];
            n_in = n_in + 1;
            @(posedge clk);
            seed = seed + 1;
        end
        i_valid <= 0; i_sol <= 0;
        repeat (8) @(posedge clk);
        $fclose(fin);
        $fclose(fout);
        $display("TB_DONE inputs=%0d outputs=%0d errors=%0d", n_in, n_out, errors);
        $finish;
    end
endmodule
