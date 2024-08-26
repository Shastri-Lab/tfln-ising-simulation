//////////////////////////////////////////////////////////////////////////////////
// Company: Korea University
// Engineer: Hyunjin Kim & Hyundo Jung
// 
// Create Date: 2022/09/19 11:00:27
// Design Name: Probabilistic Prime Factorization Machine
// Module Name: Top
// Project Name: VCBM_PA
// Target Devices: Xilinx Artix-7 (Zynq-7000)
// Tool Versions: Vivado 2020.2
// Description: Top module for prime factorization machine.
//              This module factorizes without decision block and candidate sieve.
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module Top #(
    // even number max_N_digit that is needed for formulating the machine  
    parameter [7-1:0] max_N_digit = 7'd64,
    // counter_bit means the maximum number of allowed sampling operations
    parameter [6-1:0] counter_bit = 7'd32)(
    // input seed number for measurement with LFSRs
    input [32-1:0] i_seed,
    input clk, rst,
    input i_operation_start,
    output reg operation_end,
    // operation_count means the number of sampling operations
    output reg [counter_bit-1:0] operation_count,
    input [4-1:0] i_N_MSB,
    input [max_N_digit/2-2-1:0] i_N_MID,
    input [max_N_digit/2-2-1:0] i_N_LSB,
    output reg [max_N_digit/2-1:0] X,
    output reg [max_N_digit/2-1:0] Y
    ); 
    
    // code for limiting maximum operation_count (number of sampling operations) 
    wire [counter_bit-1:0] counter_bit_minus_3 = -5;
    
    wire [max_N_digit-1:0] N = {i_N_MSB[4-1:0], i_N_MID [max_N_digit/2-2-1:0], i_N_LSB [max_N_digit/2-2-1:0]};
    reg [7-1:0] N_digit;
    
    integer k;
    
    always@(*) begin
        for (k = 0; k < max_N_digit; k = k +1) begin
            if (N[k]==1'b1) begin
                if (k - 2 * (k / 2)) N_digit <= k+1;
                else N_digit <= k+2;
            end
        end
    end
    
    wire [max_N_digit/2-1-1:0] pbit_in_0;
    wire [max_N_digit/2-1-1:0] pbit_in_1;
    wire [max_N_digit/2-1-1:0] pbit_in_2;
    wire [max_N_digit/2-1-1:0] pbit_in_3;
    wire [max_N_digit/2-1-1:0] pbit_in_4;
    wire [max_N_digit/2-1-1:0] pbit_in_5;
    wire [max_N_digit/2-1-1:0] pbit_in_6;
    wire [max_N_digit/2-1-1:0] pbit_in_7;
    wire [7:0] pbit_in [max_N_digit/2-1-1:0];
    reg [max_N_digit/2-1-1:0] pbit_out_d;
    wire [max_N_digit/2-1:0] pbit_out;
    wire [max_N_digit/2-1:0] pbit_EN;
    
    reg [3-1:0] energy_shift;
    
    wire [max_N_digit-1:0] multiply_result_XY;
    assign multiply_result_XY = X * Y;

    wire [max_N_digit/2-1:0] N_digit_decimal;
    
    genvar i, j;
    for (j = 0; j < max_N_digit/2-1; j = j +1) begin
        assign pbit_EN[0] = 1'b1;
        assign pbit_EN[j+1] = (j+1 > (N_digit/2-1)) ? 0 : 1;
        assign N_digit_decimal[0] = 1'b1;
        assign N_digit_decimal[j+1] = (j+3 > (N_digit/2-1)) ? 0 : 1;
    end

    reg [max_N_digit/2-1:0] energy_cal_target;
    reg [max_N_digit/2-1:0] energy_cal_nontarget; 

    always @(posedge clk) begin
        if ((rst && i_operation_start) == 1'b0) begin
            X <= N_digit_decimal;
            energy_cal_nontarget <= N_digit_decimal;
            Y <= N_digit_decimal;
            energy_cal_target <= N_digit_decimal;
            operation_end <= 0;
            operation_count <= 0;
            energy_shift <= 3'b111;
        end
        
        else begin
            if (operation_count[counter_bit-1:0] == counter_bit_minus_3) operation_end <= 1'b1;
            else if (energy_shift[0] == 0) begin
                if (multiply_result_XY == N) operation_end <= 1'b1;
                else begin
                    energy_cal_target <= Y;
                    energy_shift <= energy_shift + 3'b001;
                    operation_count <= operation_count + 1;
                    if (pbit_out[max_N_digit/2-1:1] == 0) begin
                        X <= N_digit_decimal;
                        energy_cal_nontarget <= {pbit_out[max_N_digit/2-1:2],2'b11};
                    end
                    else begin
                        X <=  {pbit_out[max_N_digit/2-1:1],1'b1}; 
                        energy_cal_nontarget <= {pbit_out[max_N_digit/2-1:1],1'b1};
                    end
                end
            end
            else begin
                if (multiply_result_XY == N) operation_end <= 1'b1;
                else begin
                    energy_cal_target <= X;
                    energy_shift <= energy_shift + 3'b001;
                    operation_count <= operation_count + 1;
                    if (pbit_out[max_N_digit/2-1:1] == 0) begin
                        Y <= N_digit_decimal;
                        energy_cal_nontarget <= {pbit_out[max_N_digit/2-1:2],2'b11};
                    end
                    else begin
                        Y <=  {pbit_out[max_N_digit/2-1:1],1'b1}; 
                        energy_cal_nontarget <= {pbit_out[max_N_digit/2-1:1],1'b1};
                    end
                end
            end
        end
    end
    
    for (i = 0; i < max_N_digit/2-1; i = i +1) begin
        assign pbit_in[i][7:0] = {pbit_in_7[i], pbit_in_6[i], pbit_in_5[i], pbit_in_4[i], pbit_in_3[i], pbit_in_2[i], pbit_in_1[i], pbit_in_0[i]};
        wire [48-1:0] seed_num = (128336713*i+1)*i_seed;
        wire [16-1:0] Sigmoid_out;
        wire [16-1:0] LFSR_out;
        
        Sigmoid_LUT u_Sigmoid_LUT (
            .Sigmoid_in (pbit_in[i]),
            .Sigmoid_out (Sigmoid_out)
            );
            
        LFSR_48b u_LFSR_48b (
            .clk (clk),
            .rst (rst),
            .operation_start (i_operation_start),
            .seed (seed_num),
            .LFSR_out (LFSR_out)
            );
        
        always @(*) begin
            // generates pbit output by comparing 16-bit LFSR output and 16-bit Sigmoid output
            if (LFSR_out < Sigmoid_out) pbit_out_d[i] <= 1'b1;
            else pbit_out_d[i] <= 1'b0;
        end
        
        assign pbit_out[i+1] = pbit_out_d[i] & pbit_EN[i+1];
        assign pbit_out[0] = 1;
    end
    
    Energy_calculator #(
        .max_N_digit (max_N_digit)) u_Energy_calculator (
        .X (energy_cal_target),
        .Y (energy_cal_nontarget),
        .N (N),
        .N_digit (N_digit),
        .energy_shift (energy_shift[2:1]),
        .pbit_in_0 (pbit_in_0),
        .pbit_in_1 (pbit_in_1),
        .pbit_in_2 (pbit_in_2),
        .pbit_in_3 (pbit_in_3),
        .pbit_in_4 (pbit_in_4),
        .pbit_in_5 (pbit_in_5),
        .pbit_in_6 (pbit_in_6),
        .pbit_in_7 (pbit_in_7)
    );
    
endmodule