// Top-level simulation of the Basys 3 test design: the testbench plays the PC side of the
// UART (fast baud: CLKS_PER_BIT = 8), sends 'R' or 'S', reads the reply, sends 'D' and
// writes every dumped decision to dump.txt ("<idx> <det>"), plus reply.txt.
`timescale 1ns / 1ps
module tb_top;
    parameter integer N_CELLS = 512;
    parameter ROM_FILE = "rom_small.hex";
    parameter CMD = "S";
    localparam integer CPB = 8;
    reg clk = 0, btnC = 1, rx = 1;
    wire tx;
    wire [15:0] led;
    basys3_cfar_top #(.N_CELLS(N_CELLS), .CLKS_PER_BIT(CPB), .ROM_FILE(ROM_FILE)) dut (
        .clk(clk), .btnC(btnC), .RsRx(rx), .RsTx(tx), .led(led));
    always #5 clk = ~clk;

    task send(input [7:0] b);
        integer i;
        begin
            rx = 0; repeat (CPB) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin rx = b[i]; repeat (CPB) @(posedge clk); end
            rx = 1; repeat (CPB / 4) @(posedge clk);   // return mid stop bit: the DUT may reply at once
        end
    endtask

    task recv(output [7:0] b);
        integer i;
        begin
            @(negedge tx);
            repeat (CPB + CPB / 2) @(posedge clk);
            for (i = 0; i < 8; i = i + 1) begin b[i] = tx; repeat (CPB) @(posedge clk); end
        end
    endtask

    reg [7:0] r [0:16];
    reg [7:0] hi, lo;
    integer i, n, f;
    initial begin
        repeat (10) @(posedge clk);
        btnC = 0;
        repeat (10) @(posedge clk);
        send(CMD);
        for (i = 0; i < 17; i = i + 1) recv(r[i]);
        n = {r[1], r[2]};
        f = $fopen("reply.txt", "w");
        $fdisplay(f, "%s %0d %0d %0d %0d %0d", r[0], n, {r[3], r[4], r[5], r[6]}, {r[7], r[8]}, {r[9], r[10], r[11], r[12]}, {r[13], r[14], r[15], r[16]});
        $fclose(f);
        send("D");
        f = $fopen("dump.txt", "w");
        for (i = 0; i < n; i = i + 1) begin
            recv(hi); recv(lo);
            $fdisplay(f, "%0d %0d", {hi, lo} >> 1, lo[0]);
        end
        $fclose(f);
        $display("TOP_TB_DONE n=%0d", n);
        $finish;
    end
endmodule
