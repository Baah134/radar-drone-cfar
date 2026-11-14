// Minimal 8N1 UART transmitter and receiver (CLKS_PER_BIT = f_clk / baud).
`timescale 1ns / 1ps
module uart_tx #(parameter integer CLKS_PER_BIT = 868) (
    input  wire       clk,
    input  wire       rst,
    input  wire       start,
    input  wire [7:0] data,
    output reg        tx,
    output wire       busy
);
    reg [9:0]  shreg;
    reg [3:0]  nbits;
    reg [15:0] cnt;
    assign busy = (nbits != 0);
    always @(posedge clk) begin
        if (rst) begin
            tx <= 1'b1; nbits <= 0; cnt <= 0; shreg <= 10'h3FF;
        end else if (nbits == 0) begin
            tx <= 1'b1;
            if (start) begin
                shreg <= {1'b1, data, 1'b0};
                nbits <= 4'd10;
                cnt <= 0;
            end
        end else begin
            tx <= shreg[0];
            if (cnt == CLKS_PER_BIT - 1) begin
                cnt <= 0;
                shreg <= {1'b1, shreg[9:1]};
                nbits <= nbits - 1'b1;
            end else begin
                cnt <= cnt + 1'b1;
            end
        end
    end
endmodule

module uart_rx #(parameter integer CLKS_PER_BIT = 868) (
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg        valid,
    output reg  [7:0] data
);
    reg [1:0]  sync;
    reg [3:0]  state;    // 0 idle, 1 start, 2..9 data bits, 10 stop
    reg [15:0] cnt;
    reg [7:0]  sh;
    wire r = sync[1];
    always @(posedge clk) begin
        sync <= {sync[0], rx};
        valid <= 1'b0;
        if (rst) begin
            state <= 0; cnt <= 0; sh <= 0; data <= 0;
        end else case (state)
            0: if (!r) begin state <= 1; cnt <= 0; end
            1: if (cnt == CLKS_PER_BIT / 2) begin
                   cnt <= 0;
                   state <= r ? 4'd0 : 4'd2;          // false start -> back to idle
               end else cnt <= cnt + 1'b1;
            10: if (cnt == CLKS_PER_BIT - 1) begin
                   state <= 0; cnt <= 0;
                   if (r) begin data <= sh; valid <= 1'b1; end
               end else cnt <= cnt + 1'b1;
            default: if (cnt == CLKS_PER_BIT - 1) begin
                   cnt <= 0;
                   sh <= {r, sh[7:1]};
                   state <= state + 1'b1;
               end else cnt <= cnt + 1'b1;
        endcase
    end
endmodule
