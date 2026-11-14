// 1D cell-averaging CFAR (square-law input), streaming one range cell per clock.
// Bit-exact with src/hw_model/cfar_fixed.py.
//
// Decision for cell-under-test (CUT):  cut * N * 2^ALPHA_FRAC  >  alpha_q * (lead + lag)
//   N = 2*N_TRAIN training cells, alpha_q unsigned Q(ALPHA_W-ALPHA_FRAC).ALPHA_FRAC.
// Window (W = 2*(N_TRAIN+N_GUARD)+1 cells) in a shift register sr[0]=newest .. sr[W-1]=oldest:
//   lead window sr[0 .. N_TRAIN-1], guard, CUT sr[N_TRAIN+N_GUARD], guard,
//   lag window sr[N_TRAIN+2*N_GUARD+1 .. W-1].
// Running sums: lead += in - sr[N_TRAIN-1]; lag += sr[N_TRAIN+2*N_GUARD] - sr[W-1]. They
// always equal the sums of the current register contents (also across line boundaries),
// so the data register never needs clearing; i_sol only restarts the cell counter.
// After rst the data register is NOT cleared (so synthesis can map it to SRL shift
// registers); a 1-bit 'filled' flag per position (cleared by rst) masks entries written
// before the reset out of the running-sum updates, keeping the sums exact.
// Edge rule: a decision is produced for CUT = k-(N_TRAIN+N_GUARD) once the k-th cell of the
// line (k >= W-1) has entered; cells within N_TRAIN+N_GUARD of a line end get no decision.
// Latency: the decision appears 3 clock cycles after the cell that completes its window.
// i_valid may be deasserted at any time (the pipeline advances only on valid cells).
`timescale 1ns / 1ps
module cfar_ca_1d #(
    parameter integer DATA_W     = 16,
    parameter integer N_TRAIN    = 16,   // training cells per side
    parameter integer N_GUARD    = 2,    // guard cells per side
    parameter integer ALPHA_W    = 17,   // Q5.12: alpha < 32; 17 bits keep the multiply in one DSP48 (18-bit signed port)
    parameter integer ALPHA_FRAC = 12,
    parameter integer IDX_W      = 12    // range-line index width (lines up to 4096 cells)
) (
    input  wire                     clk,
    input  wire                     rst,       // synchronous, active high
    input  wire [ALPHA_W-1:0]       alpha_q,   // must be held stable while streaming
    input  wire                     i_valid,
    input  wire                     i_sol,     // first cell of a range line
    input  wire [DATA_W-1:0]        i_data,
    output reg                      o_valid,
    output reg  [IDX_W-1:0]         o_idx,     // CUT index within the line
    output reg                      o_det
);
    localparam integer W      = 2 * (N_TRAIN + N_GUARD) + 1;
    localparam integer N      = 2 * N_TRAIN;
    localparam integer HALF   = N_TRAIN + N_GUARD;
    localparam integer SIDE_W = DATA_W + $clog2(N_TRAIN + 1);
    localparam integer SUM_W  = DATA_W + $clog2(N + 1);
    localparam integer LHS_W  = DATA_W + $clog2(N + 1) + ALPHA_FRAC;
    localparam integer RHS_W  = ALPHA_W + SUM_W;

    reg [DATA_W-1:0] sr [0:W-1];
    reg [W-1:0]      filled;     // filled[j]: sr[j] was written after the last reset
    reg [SIDE_W-1:0] lead_sum, lag_sum;
    reg [IDX_W-1:0]  k;          // index of the newest cell in the line
    integer i;

    // ---------------- stage 0: shift, running sums, cell counter
    reg              s0_valid;
    reg [IDX_W-1:0]  s0_idx;
    wire [IDX_W-1:0] k_next = i_sol ? {IDX_W{1'b0}} : k + 1'b1;
    wire [DATA_W-1:0] out_lead = filled[N_TRAIN-1]         ? sr[N_TRAIN-1]         : {DATA_W{1'b0}};
    wire [DATA_W-1:0] in_lag   = filled[N_TRAIN+2*N_GUARD] ? sr[N_TRAIN+2*N_GUARD] : {DATA_W{1'b0}};
    wire [DATA_W-1:0] out_lag  = filled[W-1]               ? sr[W-1]               : {DATA_W{1'b0}};

    // data shift register: no reset (SRL-friendly)
    always @(posedge clk) begin
        if (i_valid) begin
            sr[0] <= i_data;
            for (i = 1; i < W; i = i + 1) sr[i] <= sr[i-1];
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            filled   <= {W{1'b0}};
            lead_sum <= {SIDE_W{1'b0}};
            lag_sum  <= {SIDE_W{1'b0}};
            k        <= {IDX_W{1'b0}};
            s0_valid <= 1'b0;
            s0_idx   <= {IDX_W{1'b0}};
        end else begin
            s0_valid <= 1'b0;
            if (i_valid) begin
                filled   <= {filled[W-2:0], 1'b1};
                lead_sum <= lead_sum + i_data - out_lead;
                lag_sum  <= lag_sum + in_lag - out_lag;
                k        <= k_next;
                s0_valid <= (k_next >= W - 1);
                s0_idx   <= k_next - HALF;
            end
        end
    end

    // ---------------- stage 1: products (rhs maps to one DSP48 for SUM_W <= 25, ALPHA_W <= 18)
    reg              s1_valid;
    reg [IDX_W-1:0]  s1_idx;
    reg [LHS_W-1:0]  s1_lhs;
    reg [RHS_W-1:0]  s1_rhs;
    wire [SUM_W-1:0] sum_all = lead_sum + lag_sum;
    wire [DATA_W-1:0] cut = sr[HALF];

    always @(posedge clk) begin
        if (rst) begin
            s1_valid <= 1'b0;
            s1_idx   <= {IDX_W{1'b0}};
            s1_lhs   <= {LHS_W{1'b0}};
            s1_rhs   <= {RHS_W{1'b0}};
        end else begin
            s1_valid <= s0_valid;
            s1_idx   <= s0_idx;
            s1_lhs   <= (cut * N) << ALPHA_FRAC;
            s1_rhs   <= alpha_q * sum_all;
        end
    end

    // ---------------- stage 2: compare
    always @(posedge clk) begin
        if (rst) begin
            o_valid <= 1'b0;
            o_idx   <= {IDX_W{1'b0}};
            o_det   <= 1'b0;
        end else begin
            o_valid <= s1_valid;
            o_idx   <= s1_idx;
            o_det   <= s1_valid && (s1_lhs > s1_rhs);
        end
    end
endmodule
