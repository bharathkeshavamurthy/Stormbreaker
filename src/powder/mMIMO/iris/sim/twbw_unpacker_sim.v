// Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2018.3 (lin64) Build 2405991 Thu Dec  6 23:36:41 MST 2018
// Date        : Wed Nov 18 10:59:06 2020
// Host        : bender.ad.sklk.us running 64-bit Ubuntu 16.04.6 LTS
// Command     : write_verilog -force -mode funcsim twbw_unpacker_sim.v
// Design      : twbw_unpacker
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7z030sbg485-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* HDR_XFERS = "2" *) 
(* NotValidForBitStream *)
module twbw_unpacker
   (clk,
    rst,
    in_tctrl,
    in_tdata,
    in_tkeep,
    in_tlast,
    in_tvalid,
    in_tready,
    out_tdata,
    out_tlast,
    out_tvalid,
    out_tready);
  input clk;
  input rst;
  input [7:0]in_tctrl;
  input [63:0]in_tdata;
  input [7:0]in_tkeep;
  input in_tlast;
  input in_tvalid;
  output in_tready;
  output [63:0]out_tdata;
  output out_tlast;
  output out_tvalid;
  input out_tready;

  wire \FSM_sequential_state[0]_i_10_n_0 ;
  wire \FSM_sequential_state[0]_i_1_n_0 ;
  wire \FSM_sequential_state[0]_i_2_n_0 ;
  wire \FSM_sequential_state[0]_i_3_n_0 ;
  wire \FSM_sequential_state[0]_i_4_n_0 ;
  wire \FSM_sequential_state[0]_i_5_n_0 ;
  wire \FSM_sequential_state[0]_i_6_n_0 ;
  wire \FSM_sequential_state[0]_i_7_n_0 ;
  wire \FSM_sequential_state[0]_i_8_n_0 ;
  wire \FSM_sequential_state[0]_i_9_n_0 ;
  wire \FSM_sequential_state[1]_i_1_n_0 ;
  wire \FSM_sequential_state[1]_i_2_n_0 ;
  wire \FSM_sequential_state[2]_i_1_n_0 ;
  wire \FSM_sequential_state[2]_i_2_n_0 ;
  wire \FSM_sequential_state[2]_i_3_n_0 ;
  wire \FSM_sequential_state[2]_i_4_n_0 ;
  wire \FSM_sequential_state[2]_i_5_n_0 ;
  wire \FSM_sequential_state[2]_i_6_n_0 ;
  wire clk;
  wire [23:0]data3;
  wire data_shift;
  wire \data_shift[0]_i_1_n_0 ;
  wire \data_shift[1]_i_1_n_0 ;
  wire \data_shift[1]_i_2_n_0 ;
  wire \data_shift[1]_i_3_n_0 ;
  wire \data_shift[1]_i_4_n_0 ;
  wire \data_shift[2]_i_1_n_0 ;
  wire \data_shift[2]_i_2_n_0 ;
  wire \data_shift[2]_i_3_n_0 ;
  wire \data_shift[3]_i_1_n_0 ;
  wire \data_shift[3]_i_2_n_0 ;
  wire \data_shift[4]_i_1_n_0 ;
  wire \data_shift[4]_i_2_n_0 ;
  wire \data_shift[4]_i_3_n_0 ;
  wire \data_shift[4]_i_4_n_0 ;
  wire \data_shift[5]_i_1_n_0 ;
  wire \data_shift[5]_i_2_n_0 ;
  wire \data_shift[5]_i_3_n_0 ;
  wire \data_shift[6]_i_1_n_0 ;
  wire \data_shift[6]_i_2_n_0 ;
  wire \data_shift[7]_i_10_n_0 ;
  wire \data_shift[7]_i_2_n_0 ;
  wire \data_shift[7]_i_3_n_0 ;
  wire \data_shift[7]_i_4_n_0 ;
  wire \data_shift[7]_i_5_n_0 ;
  wire \data_shift[7]_i_6_n_0 ;
  wire \data_shift[7]_i_7_n_0 ;
  wire \data_shift[7]_i_8_n_0 ;
  wire \data_shift[7]_i_9_n_0 ;
  wire [7:1]data_shift_next;
  wire \data_shift_reg_n_0_[7] ;
  wire extra_cycle_i_10_n_0;
  wire extra_cycle_i_11_n_0;
  wire extra_cycle_i_12_n_0;
  wire extra_cycle_i_13_n_0;
  wire extra_cycle_i_14_n_0;
  wire extra_cycle_i_15_n_0;
  wire extra_cycle_i_16_n_0;
  wire extra_cycle_i_17_n_0;
  wire extra_cycle_i_18_n_0;
  wire extra_cycle_i_1_n_0;
  wire extra_cycle_i_2_n_0;
  wire extra_cycle_i_3_n_0;
  wire extra_cycle_i_4_n_0;
  wire extra_cycle_i_5_n_0;
  wire extra_cycle_i_6_n_0;
  wire extra_cycle_i_7_n_0;
  wire extra_cycle_i_8_n_0;
  wire extra_cycle_i_9_n_0;
  wire extra_cycle_reg_n_0;
  wire had_in_tlast;
  wire had_in_tlast_i_1_n_0;
  wire had_in_tlast_i_2_n_0;
  wire had_in_tlast_i_3_n_0;
  wire had_in_tlast_i_4_n_0;
  wire had_in_tlast_i_5_n_0;
  wire had_in_tlast_i_6_n_0;
  wire had_in_tlast_i_7_n_0;
  wire \hdr_shift[2]_i_1_n_0 ;
  wire [3:3]hdr_shift_next;
  wire [7:0]in_tctrl;
  wire [63:0]in_tdata;
  wire \in_tdata_prev[63]_i_1_n_0 ;
  wire in_tlast;
  wire in_tready;
  wire in_tready_INST_0_i_1_n_0;
  wire in_tready_INST_0_i_2_n_0;
  wire in_tready_INST_0_i_3_n_0;
  wire in_tready_INST_0_i_4_n_0;
  wire in_tready_INST_0_i_5_n_0;
  wire in_tready_INST_0_i_6_n_0;
  wire in_tready_INST_0_i_7_n_0;
  wire in_tready_INST_0_i_8_n_0;
  wire in_tready_INST_0_i_9_n_0;
  wire in_tvalid;
  wire [63:0]out_tdata;
  wire \out_tdata[10]_INST_0_i_1_n_0 ;
  wire \out_tdata[10]_INST_0_i_2_n_0 ;
  wire \out_tdata[10]_INST_0_i_3_n_0 ;
  wire \out_tdata[10]_INST_0_i_4_n_0 ;
  wire \out_tdata[10]_INST_0_i_5_n_0 ;
  wire \out_tdata[11]_INST_0_i_1_n_0 ;
  wire \out_tdata[11]_INST_0_i_2_n_0 ;
  wire \out_tdata[11]_INST_0_i_3_n_0 ;
  wire \out_tdata[11]_INST_0_i_4_n_0 ;
  wire \out_tdata[11]_INST_0_i_5_n_0 ;
  wire \out_tdata[11]_INST_0_i_6_n_0 ;
  wire \out_tdata[11]_INST_0_i_7_n_0 ;
  wire \out_tdata[11]_INST_0_i_8_n_0 ;
  wire \out_tdata[11]_INST_0_i_9_n_0 ;
  wire \out_tdata[12]_INST_0_i_1_n_0 ;
  wire \out_tdata[12]_INST_0_i_2_n_0 ;
  wire \out_tdata[12]_INST_0_i_3_n_0 ;
  wire \out_tdata[13]_INST_0_i_1_n_0 ;
  wire \out_tdata[13]_INST_0_i_2_n_0 ;
  wire \out_tdata[13]_INST_0_i_3_n_0 ;
  wire \out_tdata[14]_INST_0_i_1_n_0 ;
  wire \out_tdata[14]_INST_0_i_2_n_0 ;
  wire \out_tdata[14]_INST_0_i_3_n_0 ;
  wire \out_tdata[15]_INST_0_i_1_n_0 ;
  wire \out_tdata[15]_INST_0_i_2_n_0 ;
  wire \out_tdata[15]_INST_0_i_3_n_0 ;
  wire \out_tdata[15]_INST_0_i_4_n_0 ;
  wire \out_tdata[15]_INST_0_i_5_n_0 ;
  wire \out_tdata[15]_INST_0_i_6_n_0 ;
  wire \out_tdata[15]_INST_0_i_7_n_0 ;
  wire \out_tdata[15]_INST_0_i_8_n_0 ;
  wire \out_tdata[19]_INST_0_i_1_n_0 ;
  wire \out_tdata[19]_INST_0_i_2_n_0 ;
  wire \out_tdata[20]_INST_0_i_1_n_0 ;
  wire \out_tdata[20]_INST_0_i_2_n_0 ;
  wire \out_tdata[20]_INST_0_i_3_n_0 ;
  wire \out_tdata[20]_INST_0_i_4_n_0 ;
  wire \out_tdata[21]_INST_0_i_1_n_0 ;
  wire \out_tdata[21]_INST_0_i_2_n_0 ;
  wire \out_tdata[21]_INST_0_i_3_n_0 ;
  wire \out_tdata[21]_INST_0_i_4_n_0 ;
  wire \out_tdata[22]_INST_0_i_1_n_0 ;
  wire \out_tdata[22]_INST_0_i_2_n_0 ;
  wire \out_tdata[22]_INST_0_i_3_n_0 ;
  wire \out_tdata[22]_INST_0_i_4_n_0 ;
  wire \out_tdata[23]_INST_0_i_1_n_0 ;
  wire \out_tdata[23]_INST_0_i_2_n_0 ;
  wire \out_tdata[23]_INST_0_i_3_n_0 ;
  wire \out_tdata[23]_INST_0_i_4_n_0 ;
  wire \out_tdata[24]_INST_0_i_1_n_0 ;
  wire \out_tdata[24]_INST_0_i_2_n_0 ;
  wire \out_tdata[24]_INST_0_i_3_n_0 ;
  wire \out_tdata[24]_INST_0_i_4_n_0 ;
  wire \out_tdata[25]_INST_0_i_1_n_0 ;
  wire \out_tdata[25]_INST_0_i_2_n_0 ;
  wire \out_tdata[25]_INST_0_i_3_n_0 ;
  wire \out_tdata[25]_INST_0_i_4_n_0 ;
  wire \out_tdata[26]_INST_0_i_1_n_0 ;
  wire \out_tdata[26]_INST_0_i_2_n_0 ;
  wire \out_tdata[26]_INST_0_i_3_n_0 ;
  wire \out_tdata[26]_INST_0_i_4_n_0 ;
  wire \out_tdata[27]_INST_0_i_1_n_0 ;
  wire \out_tdata[27]_INST_0_i_2_n_0 ;
  wire \out_tdata[27]_INST_0_i_3_n_0 ;
  wire \out_tdata[27]_INST_0_i_4_n_0 ;
  wire \out_tdata[27]_INST_0_i_5_n_0 ;
  wire \out_tdata[27]_INST_0_i_6_n_0 ;
  wire \out_tdata[27]_INST_0_i_7_n_0 ;
  wire \out_tdata[28]_INST_0_i_1_n_0 ;
  wire \out_tdata[28]_INST_0_i_2_n_0 ;
  wire \out_tdata[28]_INST_0_i_3_n_0 ;
  wire \out_tdata[29]_INST_0_i_1_n_0 ;
  wire \out_tdata[29]_INST_0_i_2_n_0 ;
  wire \out_tdata[29]_INST_0_i_3_n_0 ;
  wire \out_tdata[30]_INST_0_i_1_n_0 ;
  wire \out_tdata[30]_INST_0_i_2_n_0 ;
  wire \out_tdata[30]_INST_0_i_3_n_0 ;
  wire \out_tdata[30]_INST_0_i_4_n_0 ;
  wire \out_tdata[30]_INST_0_i_5_n_0 ;
  wire \out_tdata[30]_INST_0_i_6_n_0 ;
  wire \out_tdata[30]_INST_0_i_7_n_0 ;
  wire \out_tdata[31]_INST_0_i_10_n_0 ;
  wire \out_tdata[31]_INST_0_i_11_n_0 ;
  wire \out_tdata[31]_INST_0_i_12_n_0 ;
  wire \out_tdata[31]_INST_0_i_13_n_0 ;
  wire \out_tdata[31]_INST_0_i_14_n_0 ;
  wire \out_tdata[31]_INST_0_i_15_n_0 ;
  wire \out_tdata[31]_INST_0_i_16_n_0 ;
  wire \out_tdata[31]_INST_0_i_17_n_0 ;
  wire \out_tdata[31]_INST_0_i_18_n_0 ;
  wire \out_tdata[31]_INST_0_i_19_n_0 ;
  wire \out_tdata[31]_INST_0_i_1_n_0 ;
  wire \out_tdata[31]_INST_0_i_2_n_0 ;
  wire \out_tdata[31]_INST_0_i_3_n_0 ;
  wire \out_tdata[31]_INST_0_i_4_n_0 ;
  wire \out_tdata[31]_INST_0_i_5_n_0 ;
  wire \out_tdata[31]_INST_0_i_6_n_0 ;
  wire \out_tdata[31]_INST_0_i_7_n_0 ;
  wire \out_tdata[31]_INST_0_i_8_n_0 ;
  wire \out_tdata[31]_INST_0_i_9_n_0 ;
  wire \out_tdata[36]_INST_0_i_1_n_0 ;
  wire \out_tdata[36]_INST_0_i_2_n_0 ;
  wire \out_tdata[37]_INST_0_i_1_n_0 ;
  wire \out_tdata[37]_INST_0_i_2_n_0 ;
  wire \out_tdata[38]_INST_0_i_1_n_0 ;
  wire \out_tdata[38]_INST_0_i_2_n_0 ;
  wire \out_tdata[39]_INST_0_i_1_n_0 ;
  wire \out_tdata[39]_INST_0_i_2_n_0 ;
  wire \out_tdata[40]_INST_0_i_1_n_0 ;
  wire \out_tdata[40]_INST_0_i_2_n_0 ;
  wire \out_tdata[41]_INST_0_i_1_n_0 ;
  wire \out_tdata[41]_INST_0_i_2_n_0 ;
  wire \out_tdata[42]_INST_0_i_1_n_0 ;
  wire \out_tdata[42]_INST_0_i_2_n_0 ;
  wire \out_tdata[43]_INST_0_i_1_n_0 ;
  wire \out_tdata[43]_INST_0_i_2_n_0 ;
  wire \out_tdata[44]_INST_0_i_1_n_0 ;
  wire \out_tdata[45]_INST_0_i_1_n_0 ;
  wire \out_tdata[46]_INST_0_i_1_n_0 ;
  wire \out_tdata[47]_INST_0_i_1_n_0 ;
  wire \out_tdata[4]_INST_0_i_1_n_0 ;
  wire \out_tdata[4]_INST_0_i_2_n_0 ;
  wire \out_tdata[4]_INST_0_i_3_n_0 ;
  wire \out_tdata[4]_INST_0_i_4_n_0 ;
  wire \out_tdata[4]_INST_0_i_5_n_0 ;
  wire \out_tdata[52]_INST_0_i_1_n_0 ;
  wire \out_tdata[52]_INST_0_i_2_n_0 ;
  wire \out_tdata[53]_INST_0_i_1_n_0 ;
  wire \out_tdata[53]_INST_0_i_2_n_0 ;
  wire \out_tdata[54]_INST_0_i_1_n_0 ;
  wire \out_tdata[54]_INST_0_i_2_n_0 ;
  wire \out_tdata[55]_INST_0_i_1_n_0 ;
  wire \out_tdata[55]_INST_0_i_2_n_0 ;
  wire \out_tdata[56]_INST_0_i_1_n_0 ;
  wire \out_tdata[56]_INST_0_i_2_n_0 ;
  wire \out_tdata[57]_INST_0_i_1_n_0 ;
  wire \out_tdata[57]_INST_0_i_2_n_0 ;
  wire \out_tdata[58]_INST_0_i_1_n_0 ;
  wire \out_tdata[58]_INST_0_i_2_n_0 ;
  wire \out_tdata[59]_INST_0_i_1_n_0 ;
  wire \out_tdata[59]_INST_0_i_2_n_0 ;
  wire \out_tdata[59]_INST_0_i_3_n_0 ;
  wire \out_tdata[59]_INST_0_i_4_n_0 ;
  wire \out_tdata[5]_INST_0_i_1_n_0 ;
  wire \out_tdata[5]_INST_0_i_2_n_0 ;
  wire \out_tdata[5]_INST_0_i_3_n_0 ;
  wire \out_tdata[5]_INST_0_i_4_n_0 ;
  wire \out_tdata[5]_INST_0_i_5_n_0 ;
  wire \out_tdata[60]_INST_0_i_1_n_0 ;
  wire \out_tdata[61]_INST_0_i_1_n_0 ;
  wire \out_tdata[62]_INST_0_i_1_n_0 ;
  wire \out_tdata[63]_INST_0_i_1_n_0 ;
  wire \out_tdata[63]_INST_0_i_2_n_0 ;
  wire \out_tdata[63]_INST_0_i_3_n_0 ;
  wire \out_tdata[63]_INST_0_i_4_n_0 ;
  wire \out_tdata[63]_INST_0_i_5_n_0 ;
  wire \out_tdata[63]_INST_0_i_6_n_0 ;
  wire \out_tdata[63]_INST_0_i_7_n_0 ;
  wire \out_tdata[6]_INST_0_i_1_n_0 ;
  wire \out_tdata[6]_INST_0_i_2_n_0 ;
  wire \out_tdata[6]_INST_0_i_3_n_0 ;
  wire \out_tdata[6]_INST_0_i_4_n_0 ;
  wire \out_tdata[6]_INST_0_i_5_n_0 ;
  wire \out_tdata[7]_INST_0_i_1_n_0 ;
  wire \out_tdata[7]_INST_0_i_2_n_0 ;
  wire \out_tdata[7]_INST_0_i_3_n_0 ;
  wire \out_tdata[7]_INST_0_i_4_n_0 ;
  wire \out_tdata[7]_INST_0_i_5_n_0 ;
  wire \out_tdata[8]_INST_0_i_1_n_0 ;
  wire \out_tdata[8]_INST_0_i_2_n_0 ;
  wire \out_tdata[8]_INST_0_i_3_n_0 ;
  wire \out_tdata[8]_INST_0_i_4_n_0 ;
  wire \out_tdata[8]_INST_0_i_5_n_0 ;
  wire \out_tdata[9]_INST_0_i_1_n_0 ;
  wire \out_tdata[9]_INST_0_i_2_n_0 ;
  wire \out_tdata[9]_INST_0_i_3_n_0 ;
  wire \out_tdata[9]_INST_0_i_4_n_0 ;
  wire \out_tdata[9]_INST_0_i_5_n_0 ;
  wire out_tlast;
  wire out_tlast_INST_0_i_1_n_0;
  wire out_tlast_INST_0_i_2_n_0;
  wire out_tlast_INST_0_i_3_n_0;
  wire out_tlast_INST_0_i_4_n_0;
  wire out_tlast_INST_0_i_5_n_0;
  wire out_tready;
  wire out_tvalid;
  wire out_tvalid_INST_0_i_1_n_0;
  wire out_tvalid_INST_0_i_2_n_0;
  wire out_tvalid_INST_0_i_3_n_0;
  wire out_tvalid_INST_0_i_4_n_0;
  wire out_tvalid_INST_0_i_5_n_0;
  wire out_tvalid_INST_0_i_6_n_0;
  wire out_tvalid_INST_0_i_7_n_0;
  wire out_tvalid_INST_0_i_8_n_0;
  wire [31:0]p_1_in;
  wire rst;
  wire [2:0]state__0;
  wire \xfer_count[0]_i_1_n_0 ;
  wire \xfer_count[0]_i_3_n_0 ;
  wire \xfer_count[0]_i_4_n_0 ;
  wire \xfer_count[0]_i_5_n_0 ;
  wire \xfer_count[0]_i_6_n_0 ;
  wire \xfer_count[0]_i_7_n_0 ;
  wire \xfer_count[12]_i_2_n_0 ;
  wire \xfer_count[12]_i_3_n_0 ;
  wire \xfer_count[12]_i_4_n_0 ;
  wire \xfer_count[12]_i_5_n_0 ;
  wire \xfer_count[4]_i_2_n_0 ;
  wire \xfer_count[4]_i_3_n_0 ;
  wire \xfer_count[4]_i_4_n_0 ;
  wire \xfer_count[4]_i_5_n_0 ;
  wire \xfer_count[8]_i_2_n_0 ;
  wire \xfer_count[8]_i_3_n_0 ;
  wire \xfer_count[8]_i_4_n_0 ;
  wire \xfer_count[8]_i_5_n_0 ;
  wire [15:0]xfer_count_reg;
  wire \xfer_count_reg[0]_i_2_n_0 ;
  wire \xfer_count_reg[0]_i_2_n_1 ;
  wire \xfer_count_reg[0]_i_2_n_2 ;
  wire \xfer_count_reg[0]_i_2_n_3 ;
  wire \xfer_count_reg[0]_i_2_n_4 ;
  wire \xfer_count_reg[0]_i_2_n_5 ;
  wire \xfer_count_reg[0]_i_2_n_6 ;
  wire \xfer_count_reg[0]_i_2_n_7 ;
  wire \xfer_count_reg[12]_i_1_n_1 ;
  wire \xfer_count_reg[12]_i_1_n_2 ;
  wire \xfer_count_reg[12]_i_1_n_3 ;
  wire \xfer_count_reg[12]_i_1_n_4 ;
  wire \xfer_count_reg[12]_i_1_n_5 ;
  wire \xfer_count_reg[12]_i_1_n_6 ;
  wire \xfer_count_reg[12]_i_1_n_7 ;
  wire \xfer_count_reg[4]_i_1_n_0 ;
  wire \xfer_count_reg[4]_i_1_n_1 ;
  wire \xfer_count_reg[4]_i_1_n_2 ;
  wire \xfer_count_reg[4]_i_1_n_3 ;
  wire \xfer_count_reg[4]_i_1_n_4 ;
  wire \xfer_count_reg[4]_i_1_n_5 ;
  wire \xfer_count_reg[4]_i_1_n_6 ;
  wire \xfer_count_reg[4]_i_1_n_7 ;
  wire \xfer_count_reg[8]_i_1_n_0 ;
  wire \xfer_count_reg[8]_i_1_n_1 ;
  wire \xfer_count_reg[8]_i_1_n_2 ;
  wire \xfer_count_reg[8]_i_1_n_3 ;
  wire \xfer_count_reg[8]_i_1_n_4 ;
  wire \xfer_count_reg[8]_i_1_n_5 ;
  wire \xfer_count_reg[8]_i_1_n_6 ;
  wire \xfer_count_reg[8]_i_1_n_7 ;
  wire [3:3]\NLW_xfer_count_reg[12]_i_1_CO_UNCONNECTED ;

  LUT6 #(
    .INIT(64'hFFFFFFFFE2EEE2E6)) 
    \FSM_sequential_state[0]_i_1 
       (.I0(state__0[0]),
        .I1(\FSM_sequential_state[2]_i_2_n_0 ),
        .I2(\FSM_sequential_state[0]_i_2_n_0 ),
        .I3(\FSM_sequential_state[0]_i_3_n_0 ),
        .I4(in_tlast),
        .I5(rst),
        .O(\FSM_sequential_state[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h3000111133551155)) 
    \FSM_sequential_state[0]_i_10 
       (.I0(state__0[0]),
        .I1(state__0[2]),
        .I2(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I3(state__0[1]),
        .I4(data_shift_next[2]),
        .I5(data_shift_next[1]),
        .O(\FSM_sequential_state[0]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFEEEAEAEA)) 
    \FSM_sequential_state[0]_i_2 
       (.I0(\FSM_sequential_state[0]_i_4_n_0 ),
        .I1(\FSM_sequential_state[0]_i_5_n_0 ),
        .I2(\FSM_sequential_state[0]_i_6_n_0 ),
        .I3(\FSM_sequential_state[0]_i_7_n_0 ),
        .I4(\data_shift[7]_i_4_n_0 ),
        .I5(\FSM_sequential_state[0]_i_8_n_0 ),
        .O(\FSM_sequential_state[0]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \FSM_sequential_state[0]_i_3 
       (.I0(state__0[1]),
        .I1(state__0[2]),
        .O(\FSM_sequential_state[0]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT4 #(
    .INIT(16'h0004)) 
    \FSM_sequential_state[0]_i_4 
       (.I0(in_tctrl[2]),
        .I1(in_tctrl[0]),
        .I2(state__0[2]),
        .I3(state__0[1]),
        .O(\FSM_sequential_state[0]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT4 #(
    .INIT(16'h0040)) 
    \FSM_sequential_state[0]_i_5 
       (.I0(had_in_tlast),
        .I1(in_tlast),
        .I2(in_tvalid),
        .I3(had_in_tlast_i_4_n_0),
        .O(\FSM_sequential_state[0]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hBAAABAAAFFFFBAAA)) 
    \FSM_sequential_state[0]_i_6 
       (.I0(in_tready_INST_0_i_5_n_0),
        .I1(state__0[0]),
        .I2(state__0[1]),
        .I3(state__0[2]),
        .I4(\FSM_sequential_state[0]_i_9_n_0 ),
        .I5(in_tready_INST_0_i_8_n_0),
        .O(\FSM_sequential_state[0]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT3 #(
    .INIT(8'h08)) 
    \FSM_sequential_state[0]_i_7 
       (.I0(data_shift_next[4]),
        .I1(data_shift_next[3]),
        .I2(data_shift_next[1]),
        .O(\FSM_sequential_state[0]_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT4 #(
    .INIT(16'h4440)) 
    \FSM_sequential_state[0]_i_8 
       (.I0(had_in_tlast_i_4_n_0),
        .I1(extra_cycle_reg_n_0),
        .I2(out_tvalid_INST_0_i_2_n_0),
        .I3(\FSM_sequential_state[0]_i_10_n_0 ),
        .O(\FSM_sequential_state[0]_i_8_n_0 ));
  LUT6 #(
    .INIT(64'h000004FF00FFFFFF)) 
    \FSM_sequential_state[0]_i_9 
       (.I0(in_tready_INST_0_i_7_n_0),
        .I1(data_shift_next[5]),
        .I2(state__0[2]),
        .I3(state__0[0]),
        .I4(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I5(state__0[1]),
        .O(\FSM_sequential_state[0]_i_9_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT4 #(
    .INIT(16'h00E2)) 
    \FSM_sequential_state[1]_i_1 
       (.I0(state__0[1]),
        .I1(\FSM_sequential_state[2]_i_2_n_0 ),
        .I2(\FSM_sequential_state[1]_i_2_n_0 ),
        .I3(rst),
        .O(\FSM_sequential_state[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT5 #(
    .INIT(32'h22202022)) 
    \FSM_sequential_state[1]_i_2 
       (.I0(\FSM_sequential_state[2]_i_4_n_0 ),
        .I1(in_tlast),
        .I2(in_tctrl[2]),
        .I3(in_tctrl[1]),
        .I4(in_tctrl[0]),
        .O(\FSM_sequential_state[1]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000022222E22)) 
    \FSM_sequential_state[2]_i_1 
       (.I0(state__0[2]),
        .I1(\FSM_sequential_state[2]_i_2_n_0 ),
        .I2(\FSM_sequential_state[2]_i_3_n_0 ),
        .I3(\FSM_sequential_state[2]_i_4_n_0 ),
        .I4(in_tlast),
        .I5(rst),
        .O(\FSM_sequential_state[2]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFF32FFFFFF32FF32)) 
    \FSM_sequential_state[2]_i_2 
       (.I0(\data_shift[7]_i_6_n_0 ),
        .I1(had_in_tlast_i_4_n_0),
        .I2(\FSM_sequential_state[2]_i_5_n_0 ),
        .I3(\FSM_sequential_state[2]_i_6_n_0 ),
        .I4(hdr_shift_next),
        .I5(\data_shift[1]_i_3_n_0 ),
        .O(\FSM_sequential_state[2]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \FSM_sequential_state[2]_i_3 
       (.I0(in_tctrl[2]),
        .I1(in_tctrl[1]),
        .I2(in_tctrl[0]),
        .O(\FSM_sequential_state[2]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT3 #(
    .INIT(8'h02)) 
    \FSM_sequential_state[2]_i_4 
       (.I0(state__0[0]),
        .I1(state__0[2]),
        .I2(state__0[1]),
        .O(\FSM_sequential_state[2]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h0000000E00000000)) 
    \FSM_sequential_state[2]_i_5 
       (.I0(data_shift_next[3]),
        .I1(data_shift_next[4]),
        .I2(data_shift_next[2]),
        .I3(data_shift_next[5]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\FSM_sequential_state[2]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h000F002200000022)) 
    \FSM_sequential_state[2]_i_6 
       (.I0(extra_cycle_i_11_n_0),
        .I1(state__0[2]),
        .I2(had_in_tlast_i_4_n_0),
        .I3(state__0[0]),
        .I4(state__0[1]),
        .I5(\data_shift[7]_i_3_n_0 ),
        .O(\FSM_sequential_state[2]_i_6_n_0 ));
  (* FSM_ENCODED_STATES = "state_data_16:010,state_data_24:011,state_data_32:100,state_data_48:101,state_data_64:110,state_drain:000,state_hdr:001" *) 
  FDRE \FSM_sequential_state_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\FSM_sequential_state[0]_i_1_n_0 ),
        .Q(state__0[0]),
        .R(1'b0));
  (* FSM_ENCODED_STATES = "state_data_16:010,state_data_24:011,state_data_32:100,state_data_48:101,state_data_64:110,state_drain:000,state_hdr:001" *) 
  FDRE \FSM_sequential_state_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\FSM_sequential_state[1]_i_1_n_0 ),
        .Q(state__0[1]),
        .R(1'b0));
  (* FSM_ENCODED_STATES = "state_data_16:010,state_data_24:011,state_data_32:100,state_data_48:101,state_data_64:110,state_drain:000,state_hdr:001" *) 
  FDRE \FSM_sequential_state_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\FSM_sequential_state[2]_i_1_n_0 ),
        .Q(state__0[2]),
        .R(1'b0));
  LUT6 #(
    .INIT(64'hFFFFFFFF00200000)) 
    \data_shift[0]_i_1 
       (.I0(\out_tdata[31]_INST_0_i_13_n_0 ),
        .I1(data_shift_next[7]),
        .I2(\data_shift_reg_n_0_[7] ),
        .I3(data_shift_next[6]),
        .I4(\out_tdata[31]_INST_0_i_14_n_0 ),
        .I5(\data_shift[7]_i_7_n_0 ),
        .O(\data_shift[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \data_shift[1]_i_1 
       (.I0(\data_shift[1]_i_2_n_0 ),
        .I1(\FSM_sequential_state[0]_i_3_n_0 ),
        .I2(data_shift_next[1]),
        .I3(\data_shift[1]_i_3_n_0 ),
        .I4(\data_shift[1]_i_4_n_0 ),
        .I5(data_shift_next[2]),
        .O(\data_shift[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT5 #(
    .INIT(32'h00000C0A)) 
    \data_shift[1]_i_2 
       (.I0(\data_shift[4]_i_4_n_0 ),
        .I1(\data_shift[2]_i_3_n_0 ),
        .I2(data_shift_next[3]),
        .I3(data_shift_next[4]),
        .I4(data_shift_next[2]),
        .O(\data_shift[1]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h000000A800000000)) 
    \data_shift[1]_i_3 
       (.I0(out_tready),
        .I1(had_in_tlast),
        .I2(in_tvalid),
        .I3(state__0[1]),
        .I4(state__0[2]),
        .I5(state__0[0]),
        .O(\data_shift[1]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \data_shift[1]_i_4 
       (.I0(state__0[2]),
        .I1(state__0[0]),
        .O(\data_shift[1]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAABBAAABAABAAAAA)) 
    \data_shift[2]_i_1 
       (.I0(\data_shift[2]_i_2_n_0 ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[4]),
        .I3(data_shift_next[3]),
        .I4(\data_shift[2]_i_3_n_0 ),
        .I5(\data_shift[4]_i_4_n_0 ),
        .O(\data_shift[2]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAAACAAA0AAA0AAA0)) 
    \data_shift[2]_i_2 
       (.I0(data_shift_next[2]),
        .I1(state__0[0]),
        .I2(state__0[2]),
        .I3(state__0[1]),
        .I4(\data_shift[7]_i_3_n_0 ),
        .I5(out_tready),
        .O(\data_shift[2]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT3 #(
    .INIT(8'hA4)) 
    \data_shift[2]_i_3 
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .O(\data_shift[2]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT5 #(
    .INIT(32'hFFFFFFE0)) 
    \data_shift[3]_i_1 
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .I2(data_shift_next[3]),
        .I3(\data_shift[3]_i_2_n_0 ),
        .I4(\data_shift[4]_i_2_n_0 ),
        .O(\data_shift[3]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hA4A40000FF000000)) 
    \data_shift[3]_i_2 
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(\data_shift[4]_i_4_n_0 ),
        .I4(\data_shift[4]_i_3_n_0 ),
        .I5(data_shift_next[4]),
        .O(\data_shift[3]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hEAEAFFEAEAEAEAEA)) 
    \data_shift[4]_i_1 
       (.I0(\data_shift[4]_i_2_n_0 ),
        .I1(\FSM_sequential_state[0]_i_3_n_0 ),
        .I2(data_shift_next[4]),
        .I3(\data_shift[4]_i_3_n_0 ),
        .I4(data_shift_next[3]),
        .I5(\data_shift[4]_i_4_n_0 ),
        .O(\data_shift[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT5 #(
    .INIT(32'hAAAAAEAA)) 
    \data_shift[4]_i_2 
       (.I0(\data_shift[1]_i_3_n_0 ),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(state__0[2]),
        .I4(state__0[0]),
        .O(\data_shift[4]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT2 #(
    .INIT(4'h1)) 
    \data_shift[4]_i_3 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .O(\data_shift[4]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT5 #(
    .INIT(32'h00000020)) 
    \data_shift[4]_i_4 
       (.I0(\out_tdata[31]_INST_0_i_14_n_0 ),
        .I1(data_shift_next[6]),
        .I2(\data_shift_reg_n_0_[7] ),
        .I3(data_shift_next[7]),
        .I4(data_shift_next[5]),
        .O(\data_shift[4]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFEAFFEAFFEAEAEA)) 
    \data_shift[5]_i_1 
       (.I0(\data_shift[7]_i_7_n_0 ),
        .I1(\data_shift[5]_i_2_n_0 ),
        .I2(\data_shift[5]_i_3_n_0 ),
        .I3(data_shift_next[5]),
        .I4(state__0[1]),
        .I5(state__0[2]),
        .O(\data_shift[5]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000004000000)) 
    \data_shift[5]_i_2 
       (.I0(data_shift_next[7]),
        .I1(\data_shift_reg_n_0_[7] ),
        .I2(data_shift_next[6]),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .I5(state__0[2]),
        .O(\data_shift[5]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT4 #(
    .INIT(16'h0001)) 
    \data_shift[5]_i_3 
       (.I0(data_shift_next[3]),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(data_shift_next[4]),
        .O(\data_shift[5]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFEAAAEAAAEAAA)) 
    \data_shift[6]_i_1 
       (.I0(\data_shift[7]_i_7_n_0 ),
        .I1(\data_shift[6]_i_2_n_0 ),
        .I2(\out_tdata[31]_INST_0_i_14_n_0 ),
        .I3(\out_tdata[31]_INST_0_i_13_n_0 ),
        .I4(data_shift_next[6]),
        .I5(\FSM_sequential_state[0]_i_3_n_0 ),
        .O(\data_shift[6]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h2)) 
    \data_shift[6]_i_2 
       (.I0(\data_shift_reg_n_0_[7] ),
        .I1(data_shift_next[7]),
        .O(\data_shift[6]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFF08FF00FF00)) 
    \data_shift[7]_i_1 
       (.I0(\data_shift[7]_i_3_n_0 ),
        .I1(\data_shift[7]_i_4_n_0 ),
        .I2(hdr_shift_next),
        .I3(\data_shift[7]_i_5_n_0 ),
        .I4(\data_shift[7]_i_6_n_0 ),
        .I5(out_tready),
        .O(data_shift));
  LUT6 #(
    .INIT(64'h0000EEFEEEFE0000)) 
    \data_shift[7]_i_10 
       (.I0(had_in_tlast),
        .I1(in_tvalid),
        .I2(data_shift_next[1]),
        .I3(data_shift_next[2]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\data_shift[7]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFEAAAEAAAEAAA)) 
    \data_shift[7]_i_2 
       (.I0(\data_shift[7]_i_7_n_0 ),
        .I1(\data_shift_reg_n_0_[7] ),
        .I2(\data_shift[7]_i_8_n_0 ),
        .I3(\out_tdata[31]_INST_0_i_13_n_0 ),
        .I4(data_shift_next[7]),
        .I5(\FSM_sequential_state[0]_i_3_n_0 ),
        .O(\data_shift[7]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \data_shift[7]_i_3 
       (.I0(had_in_tlast),
        .I1(in_tvalid),
        .O(\data_shift[7]_i_3_n_0 ));
  LUT2 #(
    .INIT(4'h2)) 
    \data_shift[7]_i_4 
       (.I0(state__0[0]),
        .I1(state__0[1]),
        .O(\data_shift[7]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000E000)) 
    \data_shift[7]_i_5 
       (.I0(data_shift_next[3]),
        .I1(data_shift_next[4]),
        .I2(\out_tdata[31]_INST_0_i_19_n_0 ),
        .I3(out_tready),
        .I4(data_shift_next[2]),
        .I5(data_shift_next[5]),
        .O(\data_shift[7]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFEEFEAAAA)) 
    \data_shift[7]_i_6 
       (.I0(out_tvalid_INST_0_i_3_n_0),
        .I1(in_tready_INST_0_i_7_n_0),
        .I2(data_shift_next[2]),
        .I3(data_shift_next[5]),
        .I4(\out_tdata[31]_INST_0_i_14_n_0 ),
        .I5(\data_shift[7]_i_9_n_0 ),
        .O(\data_shift[7]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFF01000000)) 
    \data_shift[7]_i_7 
       (.I0(data_shift_next[3]),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(data_shift_next[4]),
        .I4(\data_shift[2]_i_3_n_0 ),
        .I5(\data_shift[4]_i_2_n_0 ),
        .O(\data_shift[7]_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT4 #(
    .INIT(16'h0040)) 
    \data_shift[7]_i_8 
       (.I0(state__0[2]),
        .I1(state__0[0]),
        .I2(state__0[1]),
        .I3(data_shift_next[6]),
        .O(\data_shift[7]_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFF0AC00000)) 
    \data_shift[7]_i_9 
       (.I0(data_shift_next[5]),
        .I1(state__0[0]),
        .I2(state__0[2]),
        .I3(state__0[1]),
        .I4(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I5(\data_shift[7]_i_10_n_0 ),
        .O(\data_shift[7]_i_9_n_0 ));
  FDSE \data_shift_reg[0] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[0]_i_1_n_0 ),
        .Q(data_shift_next[1]),
        .S(rst));
  FDSE \data_shift_reg[1] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[1]_i_1_n_0 ),
        .Q(data_shift_next[2]),
        .S(rst));
  FDSE \data_shift_reg[2] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[2]_i_1_n_0 ),
        .Q(data_shift_next[3]),
        .S(rst));
  FDSE \data_shift_reg[3] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[3]_i_1_n_0 ),
        .Q(data_shift_next[4]),
        .S(rst));
  FDSE \data_shift_reg[4] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[4]_i_1_n_0 ),
        .Q(data_shift_next[5]),
        .S(rst));
  FDSE \data_shift_reg[5] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[5]_i_1_n_0 ),
        .Q(data_shift_next[6]),
        .S(rst));
  FDSE \data_shift_reg[6] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[6]_i_1_n_0 ),
        .Q(data_shift_next[7]),
        .S(rst));
  FDSE \data_shift_reg[7] 
       (.C(clk),
        .CE(data_shift),
        .D(\data_shift[7]_i_2_n_0 ),
        .Q(\data_shift_reg_n_0_[7] ),
        .S(rst));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFEEFEEE)) 
    extra_cycle_i_1
       (.I0(extra_cycle_i_2_n_0),
        .I1(extra_cycle_i_3_n_0),
        .I2(extra_cycle_i_4_n_0),
        .I3(extra_cycle_i_5_n_0),
        .I4(extra_cycle_i_6_n_0),
        .I5(extra_cycle_i_7_n_0),
        .O(extra_cycle_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT3 #(
    .INIT(8'h01)) 
    extra_cycle_i_10
       (.I0(in_tvalid),
        .I1(had_in_tlast),
        .I2(rst),
        .O(extra_cycle_i_10_n_0));
  LUT2 #(
    .INIT(4'h8)) 
    extra_cycle_i_11
       (.I0(in_tlast),
        .I1(in_tvalid),
        .O(extra_cycle_i_11_n_0));
  LUT6 #(
    .INIT(64'h000A000A00FF000E)) 
    extra_cycle_i_12
       (.I0(extra_cycle_i_16_n_0),
        .I1(state__0[1]),
        .I2(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I3(in_tready_INST_0_i_8_n_0),
        .I4(state__0[2]),
        .I5(state__0[0]),
        .O(extra_cycle_i_12_n_0));
  LUT6 #(
    .INIT(64'h07C037C007C007C0)) 
    extra_cycle_i_13
       (.I0(extra_cycle_i_17_n_0),
        .I1(state__0[2]),
        .I2(state__0[1]),
        .I3(state__0[0]),
        .I4(in_tready_INST_0_i_7_n_0),
        .I5(extra_cycle_i_18_n_0),
        .O(extra_cycle_i_13_n_0));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'h08000000)) 
    extra_cycle_i_14
       (.I0(state__0[0]),
        .I1(state__0[2]),
        .I2(state__0[1]),
        .I3(data_shift_next[4]),
        .I4(data_shift_next[3]),
        .O(extra_cycle_i_14_n_0));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT2 #(
    .INIT(4'h2)) 
    extra_cycle_i_15
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .O(extra_cycle_i_15_n_0));
  LUT6 #(
    .INIT(64'h0800000000000000)) 
    extra_cycle_i_16
       (.I0(state__0[1]),
        .I1(data_shift_next[5]),
        .I2(state__0[2]),
        .I3(\data_shift_reg_n_0_[7] ),
        .I4(data_shift_next[7]),
        .I5(data_shift_next[6]),
        .O(extra_cycle_i_16_n_0));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT4 #(
    .INIT(16'h4FFF)) 
    extra_cycle_i_17
       (.I0(data_shift_next[2]),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[4]),
        .I3(data_shift_next[3]),
        .O(extra_cycle_i_17_n_0));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'h00080001)) 
    extra_cycle_i_18
       (.I0(data_shift_next[5]),
        .I1(data_shift_next[3]),
        .I2(data_shift_next[1]),
        .I3(data_shift_next[2]),
        .I4(data_shift_next[4]),
        .O(extra_cycle_i_18_n_0));
  LUT6 #(
    .INIT(64'h00A800AB00000000)) 
    extra_cycle_i_2
       (.I0(had_in_tlast_i_4_n_0),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(rst),
        .I4(state__0[0]),
        .I5(extra_cycle_i_8_n_0),
        .O(extra_cycle_i_2_n_0));
  LUT6 #(
    .INIT(64'h0C00008800000088)) 
    extra_cycle_i_3
       (.I0(\out_tdata[31]_INST_0_i_17_n_0 ),
        .I1(extra_cycle_i_5_n_0),
        .I2(state__0[0]),
        .I3(data_shift_next[1]),
        .I4(data_shift_next[2]),
        .I5(extra_cycle_i_9_n_0),
        .O(extra_cycle_i_3_n_0));
  LUT6 #(
    .INIT(64'h0001000000000000)) 
    extra_cycle_i_4
       (.I0(in_tready_INST_0_i_7_n_0),
        .I1(state__0[2]),
        .I2(in_tready_INST_0_i_8_n_0),
        .I3(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I4(state__0[1]),
        .I5(data_shift_next[5]),
        .O(extra_cycle_i_4_n_0));
  LUT6 #(
    .INIT(64'hA888A888A8888888)) 
    extra_cycle_i_5
       (.I0(extra_cycle_i_10_n_0),
        .I1(extra_cycle_reg_n_0),
        .I2(in_tready_INST_0_i_3_n_0),
        .I3(extra_cycle_i_11_n_0),
        .I4(extra_cycle_i_12_n_0),
        .I5(extra_cycle_i_13_n_0),
        .O(extra_cycle_i_5_n_0));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT3 #(
    .INIT(8'h40)) 
    extra_cycle_i_6
       (.I0(state__0[0]),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .O(extra_cycle_i_6_n_0));
  LUT6 #(
    .INIT(64'hFFFF0000F4440000)) 
    extra_cycle_i_7
       (.I0(data_shift_next[1]),
        .I1(extra_cycle_i_14_n_0),
        .I2(extra_cycle_i_15_n_0),
        .I3(\out_tdata[11]_INST_0_i_8_n_0 ),
        .I4(extra_cycle_i_5_n_0),
        .I5(\out_tdata[31]_INST_0_i_7_n_0 ),
        .O(extra_cycle_i_7_n_0));
  LUT6 #(
    .INIT(64'hFFFFFFFF00E00000)) 
    extra_cycle_i_8
       (.I0(extra_cycle_i_13_n_0),
        .I1(extra_cycle_i_12_n_0),
        .I2(extra_cycle_i_11_n_0),
        .I3(had_in_tlast),
        .I4(out_tready),
        .I5(extra_cycle_reg_n_0),
        .O(extra_cycle_i_8_n_0));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT3 #(
    .INIT(8'h8F)) 
    extra_cycle_i_9
       (.I0(data_shift_next[4]),
        .I1(data_shift_next[3]),
        .I2(state__0[1]),
        .O(extra_cycle_i_9_n_0));
  FDRE extra_cycle_reg
       (.C(clk),
        .CE(1'b1),
        .D(extra_cycle_i_1_n_0),
        .Q(extra_cycle_reg_n_0),
        .R(1'b0));
  LUT6 #(
    .INIT(64'hAAEEAAEEAAEAAAAA)) 
    had_in_tlast_i_1
       (.I0(had_in_tlast_i_2_n_0),
        .I1(had_in_tlast_i_3_n_0),
        .I2(\FSM_sequential_state[0]_i_3_n_0 ),
        .I3(rst),
        .I4(had_in_tlast_i_4_n_0),
        .I5(had_in_tlast_i_5_n_0),
        .O(had_in_tlast_i_1_n_0));
  LUT6 #(
    .INIT(64'hFFF8000000000000)) 
    had_in_tlast_i_2
       (.I0(in_tready_INST_0_i_2_n_0),
        .I1(extra_cycle_i_11_n_0),
        .I2(had_in_tlast_i_6_n_0),
        .I3(had_in_tlast),
        .I4(extra_cycle_i_10_n_0),
        .I5(had_in_tlast_i_7_n_0),
        .O(had_in_tlast_i_2_n_0));
  LUT6 #(
    .INIT(64'hFFFFFFFF888888C8)) 
    had_in_tlast_i_3
       (.I0(in_tready_INST_0_i_2_n_0),
        .I1(extra_cycle_i_11_n_0),
        .I2(in_tready_INST_0_i_1_n_0),
        .I3(data_shift_next[1]),
        .I4(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I5(had_in_tlast),
        .O(had_in_tlast_i_3_n_0));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT5 #(
    .INIT(32'hFFFEFFFF)) 
    had_in_tlast_i_4
       (.I0(out_tlast_INST_0_i_5_n_0),
        .I1(out_tlast_INST_0_i_4_n_0),
        .I2(out_tlast_INST_0_i_3_n_0),
        .I3(out_tlast_INST_0_i_2_n_0),
        .I4(out_tready),
        .O(had_in_tlast_i_4_n_0));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT5 #(
    .INIT(32'h50505070)) 
    had_in_tlast_i_5
       (.I0(out_tready),
        .I1(in_tlast),
        .I2(state__0[0]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .O(had_in_tlast_i_5_n_0));
  LUT6 #(
    .INIT(64'h0080000000000000)) 
    had_in_tlast_i_6
       (.I0(in_tlast),
        .I1(in_tvalid),
        .I2(in_tready_INST_0_i_1_n_0),
        .I3(data_shift_next[1]),
        .I4(data_shift_next[3]),
        .I5(data_shift_next[4]),
        .O(had_in_tlast_i_6_n_0));
  LUT6 #(
    .INIT(64'hFFFFFFFFBAAEBEAE)) 
    had_in_tlast_i_7
       (.I0(in_tready_INST_0_i_5_n_0),
        .I1(state__0[0]),
        .I2(state__0[1]),
        .I3(state__0[2]),
        .I4(extra_cycle_i_17_n_0),
        .I5(extra_cycle_i_12_n_0),
        .O(had_in_tlast_i_7_n_0));
  FDRE had_in_tlast_reg
       (.C(clk),
        .CE(1'b1),
        .D(had_in_tlast_i_1_n_0),
        .Q(had_in_tlast),
        .R(1'b0));
  LUT3 #(
    .INIT(8'hF6)) 
    \hdr_shift[2]_i_1 
       (.I0(hdr_shift_next),
        .I1(\data_shift[1]_i_3_n_0 ),
        .I2(rst),
        .O(\hdr_shift[2]_i_1_n_0 ));
  FDRE \hdr_shift_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\hdr_shift[2]_i_1_n_0 ),
        .Q(hdr_shift_next),
        .R(1'b0));
  LUT6 #(
    .INIT(64'hFFFF000040000000)) 
    \in_tdata_prev[63]_i_1 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[4]),
        .I2(data_shift_next[3]),
        .I3(in_tready_INST_0_i_1_n_0),
        .I4(in_tvalid),
        .I5(in_tready_INST_0_i_2_n_0),
        .O(\in_tdata_prev[63]_i_1_n_0 ));
  FDRE \in_tdata_prev_reg[10] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[10]),
        .Q(data3[2]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[11] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[11]),
        .Q(data3[3]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[12] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[12]),
        .Q(data3[4]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[13] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[13]),
        .Q(data3[5]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[14] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[14]),
        .Q(data3[6]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[15] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[15]),
        .Q(data3[7]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[16] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[16]),
        .Q(data3[8]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[17] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[17]),
        .Q(data3[9]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[18] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[18]),
        .Q(data3[10]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[19] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[19]),
        .Q(data3[11]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[20] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[20]),
        .Q(data3[12]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[21] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[21]),
        .Q(data3[13]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[22] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[22]),
        .Q(data3[14]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[23] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[23]),
        .Q(data3[15]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[24] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[24]),
        .Q(data3[16]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[25] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[25]),
        .Q(data3[17]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[26] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[26]),
        .Q(data3[18]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[27] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[27]),
        .Q(data3[19]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[28] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[28]),
        .Q(data3[20]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[29] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[29]),
        .Q(data3[21]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[30] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[30]),
        .Q(data3[22]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[31] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[31]),
        .Q(data3[23]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[32] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[32]),
        .Q(p_1_in[0]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[33] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[33]),
        .Q(p_1_in[1]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[34] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[34]),
        .Q(p_1_in[2]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[35] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[35]),
        .Q(p_1_in[3]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[36] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[36]),
        .Q(p_1_in[4]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[37] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[37]),
        .Q(p_1_in[5]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[38] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[38]),
        .Q(p_1_in[6]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[39] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[39]),
        .Q(p_1_in[7]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[40] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[40]),
        .Q(p_1_in[8]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[41] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[41]),
        .Q(p_1_in[9]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[42] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[42]),
        .Q(p_1_in[10]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[43] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[43]),
        .Q(p_1_in[11]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[44] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[44]),
        .Q(p_1_in[12]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[45] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[45]),
        .Q(p_1_in[13]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[46] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[46]),
        .Q(p_1_in[14]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[47] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[47]),
        .Q(p_1_in[15]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[48] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[48]),
        .Q(p_1_in[16]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[49] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[49]),
        .Q(p_1_in[17]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[50] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[50]),
        .Q(p_1_in[18]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[51] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[51]),
        .Q(p_1_in[19]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[52] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[52]),
        .Q(p_1_in[20]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[53] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[53]),
        .Q(p_1_in[21]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[54] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[54]),
        .Q(p_1_in[22]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[55] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[55]),
        .Q(p_1_in[23]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[56] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[56]),
        .Q(p_1_in[24]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[57] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[57]),
        .Q(p_1_in[25]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[58] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[58]),
        .Q(p_1_in[26]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[59] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[59]),
        .Q(p_1_in[27]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[60] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[60]),
        .Q(p_1_in[28]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[61] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[61]),
        .Q(p_1_in[29]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[62] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[62]),
        .Q(p_1_in[30]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[63] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[63]),
        .Q(p_1_in[31]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[8] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[8]),
        .Q(data3[0]),
        .R(1'b0));
  FDRE \in_tdata_prev_reg[9] 
       (.C(clk),
        .CE(\in_tdata_prev[63]_i_1_n_0 ),
        .D(in_tdata[9]),
        .Q(data3[1]),
        .R(1'b0));
  LUT5 #(
    .INIT(32'hFFFF0800)) 
    in_tready_INST_0
       (.I0(data_shift_next[4]),
        .I1(data_shift_next[3]),
        .I2(data_shift_next[1]),
        .I3(in_tready_INST_0_i_1_n_0),
        .I4(in_tready_INST_0_i_2_n_0),
        .O(in_tready));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT4 #(
    .INIT(16'h0400)) 
    in_tready_INST_0_i_1
       (.I0(state__0[1]),
        .I1(state__0[0]),
        .I2(had_in_tlast),
        .I3(out_tready),
        .O(in_tready_INST_0_i_1_n_0));
  LUT6 #(
    .INIT(64'hFF01FF01FF011901)) 
    in_tready_INST_0_i_2
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(in_tready_INST_0_i_3_n_0),
        .I4(in_tready_INST_0_i_4_n_0),
        .I5(in_tready_INST_0_i_5_n_0),
        .O(in_tready_INST_0_i_2_n_0));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT2 #(
    .INIT(4'h2)) 
    in_tready_INST_0_i_3
       (.I0(out_tready),
        .I1(had_in_tlast),
        .O(in_tready_INST_0_i_3_n_0));
  LUT6 #(
    .INIT(64'h0000000017173717)) 
    in_tready_INST_0_i_4
       (.I0(state__0[1]),
        .I1(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I2(state__0[0]),
        .I3(in_tready_INST_0_i_6_n_0),
        .I4(in_tready_INST_0_i_7_n_0),
        .I5(in_tready_INST_0_i_8_n_0),
        .O(in_tready_INST_0_i_4_n_0));
  LUT6 #(
    .INIT(64'h1000000000000100)) 
    in_tready_INST_0_i_5
       (.I0(in_tready_INST_0_i_9_n_0),
        .I1(in_tready_INST_0_i_7_n_0),
        .I2(data_shift_next[4]),
        .I3(\data_shift[4]_i_3_n_0 ),
        .I4(data_shift_next[3]),
        .I5(data_shift_next[5]),
        .O(in_tready_INST_0_i_5_n_0));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT2 #(
    .INIT(4'h2)) 
    in_tready_INST_0_i_6
       (.I0(data_shift_next[5]),
        .I1(state__0[2]),
        .O(in_tready_INST_0_i_6_n_0));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT3 #(
    .INIT(8'h7F)) 
    in_tready_INST_0_i_7
       (.I0(\data_shift_reg_n_0_[7] ),
        .I1(data_shift_next[7]),
        .I2(data_shift_next[6]),
        .O(in_tready_INST_0_i_7_n_0));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT2 #(
    .INIT(4'h7)) 
    in_tready_INST_0_i_8
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .O(in_tready_INST_0_i_8_n_0));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT2 #(
    .INIT(4'hB)) 
    in_tready_INST_0_i_9
       (.I0(state__0[2]),
        .I1(state__0[0]),
        .O(in_tready_INST_0_i_9_n_0));
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[0]_INST_0 
       (.I0(p_1_in[0]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(in_tdata[0]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(out_tdata[0]));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFEFEFE)) 
    \out_tdata[10]_INST_0 
       (.I0(\out_tdata[10]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[10]_INST_0_i_2_n_0 ),
        .I2(\out_tdata[10]_INST_0_i_3_n_0 ),
        .I3(p_1_in[22]),
        .I4(\out_tdata[11]_INST_0_i_4_n_0 ),
        .I5(\out_tdata[10]_INST_0_i_4_n_0 ),
        .O(out_tdata[10]));
  LUT4 #(
    .INIT(16'hEEEA)) 
    \out_tdata[10]_INST_0_i_1 
       (.I0(\out_tdata[10]_INST_0_i_5_n_0 ),
        .I1(p_1_in[6]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[10]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT3 #(
    .INIT(8'hE0)) 
    \out_tdata[10]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I2(data3[14]),
        .O(\out_tdata[10]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[10]_INST_0_i_3 
       (.I0(in_tdata[6]),
        .I1(\out_tdata[11]_INST_0_i_7_n_0 ),
        .I2(in_tdata[10]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(\out_tdata[10]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[10]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(p_1_in[30]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[6]),
        .I4(data3[22]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[10]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[10]_INST_0_i_5 
       (.I0(p_1_in[10]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(p_1_in[14]),
        .I3(\out_tdata[15]_INST_0_i_6_n_0 ),
        .O(\out_tdata[10]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFEFEFE)) 
    \out_tdata[11]_INST_0 
       (.I0(\out_tdata[11]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[11]_INST_0_i_2_n_0 ),
        .I2(\out_tdata[11]_INST_0_i_3_n_0 ),
        .I3(p_1_in[23]),
        .I4(\out_tdata[11]_INST_0_i_4_n_0 ),
        .I5(\out_tdata[11]_INST_0_i_5_n_0 ),
        .O(out_tdata[11]));
  LUT4 #(
    .INIT(16'hEEEA)) 
    \out_tdata[11]_INST_0_i_1 
       (.I0(\out_tdata[11]_INST_0_i_6_n_0 ),
        .I1(p_1_in[7]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[11]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT3 #(
    .INIT(8'hE0)) 
    \out_tdata[11]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I2(data3[15]),
        .O(\out_tdata[11]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[11]_INST_0_i_3 
       (.I0(in_tdata[7]),
        .I1(\out_tdata[11]_INST_0_i_7_n_0 ),
        .I2(in_tdata[11]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(\out_tdata[11]_INST_0_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \out_tdata[11]_INST_0_i_4 
       (.I0(\out_tdata[30]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[11]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[11]_INST_0_i_5 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(p_1_in[31]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[7]),
        .I4(data3[23]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[11]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[11]_INST_0_i_6 
       (.I0(p_1_in[11]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(p_1_in[15]),
        .I3(\out_tdata[15]_INST_0_i_6_n_0 ),
        .O(\out_tdata[11]_INST_0_i_6_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFF0080)) 
    \out_tdata[11]_INST_0_i_7 
       (.I0(data_shift_next[5]),
        .I1(state__0[1]),
        .I2(\out_tdata[11]_INST_0_i_8_n_0 ),
        .I3(\out_tdata[11]_INST_0_i_9_n_0 ),
        .I4(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I5(\out_tdata[27]_INST_0_i_6_n_0 ),
        .O(\out_tdata[11]_INST_0_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \out_tdata[11]_INST_0_i_8 
       (.I0(data_shift_next[2]),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[4]),
        .I3(data_shift_next[3]),
        .O(\out_tdata[11]_INST_0_i_8_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT4 #(
    .INIT(16'hFF7F)) 
    \out_tdata[11]_INST_0_i_9 
       (.I0(data_shift_next[6]),
        .I1(data_shift_next[7]),
        .I2(\data_shift_reg_n_0_[7] ),
        .I3(state__0[2]),
        .O(\out_tdata[11]_INST_0_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFEA)) 
    \out_tdata[12]_INST_0 
       (.I0(\out_tdata[12]_INST_0_i_1_n_0 ),
        .I1(data3[16]),
        .I2(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I3(\out_tdata[15]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[12]_INST_0_i_2_n_0 ),
        .I5(\out_tdata[12]_INST_0_i_3_n_0 ),
        .O(out_tdata[12]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[12]_INST_0_i_1 
       (.I0(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I1(p_1_in[12]),
        .I2(\out_tdata[31]_INST_0_i_11_n_0 ),
        .I3(p_1_in[8]),
        .I4(p_1_in[24]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[12]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[12]_INST_0_i_2 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[12]),
        .I2(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I3(in_tdata[0]),
        .I4(in_tdata[8]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[12]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[12]_INST_0_i_3 
       (.I0(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I1(data3[8]),
        .I2(\out_tdata[15]_INST_0_i_6_n_0 ),
        .I3(p_1_in[16]),
        .I4(p_1_in[0]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[12]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFEA)) 
    \out_tdata[13]_INST_0 
       (.I0(\out_tdata[13]_INST_0_i_1_n_0 ),
        .I1(data3[17]),
        .I2(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I3(\out_tdata[15]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[13]_INST_0_i_2_n_0 ),
        .I5(\out_tdata[13]_INST_0_i_3_n_0 ),
        .O(out_tdata[13]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[13]_INST_0_i_1 
       (.I0(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I1(p_1_in[13]),
        .I2(\out_tdata[31]_INST_0_i_11_n_0 ),
        .I3(p_1_in[9]),
        .I4(p_1_in[25]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[13]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[13]_INST_0_i_2 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[13]),
        .I2(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I3(in_tdata[1]),
        .I4(in_tdata[9]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[13]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[13]_INST_0_i_3 
       (.I0(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I1(data3[9]),
        .I2(\out_tdata[15]_INST_0_i_6_n_0 ),
        .I3(p_1_in[17]),
        .I4(p_1_in[1]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[13]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFEA)) 
    \out_tdata[14]_INST_0 
       (.I0(\out_tdata[14]_INST_0_i_1_n_0 ),
        .I1(data3[18]),
        .I2(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I3(\out_tdata[15]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[14]_INST_0_i_2_n_0 ),
        .I5(\out_tdata[14]_INST_0_i_3_n_0 ),
        .O(out_tdata[14]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[14]_INST_0_i_1 
       (.I0(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I1(p_1_in[14]),
        .I2(\out_tdata[31]_INST_0_i_11_n_0 ),
        .I3(p_1_in[10]),
        .I4(p_1_in[26]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[14]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[14]_INST_0_i_2 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[14]),
        .I2(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I3(in_tdata[2]),
        .I4(in_tdata[10]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[14]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[14]_INST_0_i_3 
       (.I0(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I1(data3[10]),
        .I2(\out_tdata[15]_INST_0_i_6_n_0 ),
        .I3(p_1_in[18]),
        .I4(p_1_in[2]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[14]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFEA)) 
    \out_tdata[15]_INST_0 
       (.I0(\out_tdata[15]_INST_0_i_1_n_0 ),
        .I1(data3[19]),
        .I2(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I3(\out_tdata[15]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[15]_INST_0_i_3_n_0 ),
        .I5(\out_tdata[15]_INST_0_i_4_n_0 ),
        .O(out_tdata[15]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[15]_INST_0_i_1 
       (.I0(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I1(p_1_in[15]),
        .I2(\out_tdata[31]_INST_0_i_11_n_0 ),
        .I3(p_1_in[11]),
        .I4(p_1_in[27]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[15]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT3 #(
    .INIT(8'hF8)) 
    \out_tdata[15]_INST_0_i_2 
       (.I0(p_1_in[23]),
        .I1(\out_tdata[30]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[15]_INST_0_i_5_n_0 ),
        .O(\out_tdata[15]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[15]_INST_0_i_3 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[15]),
        .I2(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I3(in_tdata[3]),
        .I4(in_tdata[11]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[15]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[15]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I1(data3[11]),
        .I2(\out_tdata[15]_INST_0_i_6_n_0 ),
        .I3(p_1_in[19]),
        .I4(p_1_in[3]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[15]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hCC00F0000000AA00)) 
    \out_tdata[15]_INST_0_i_5 
       (.I0(p_1_in[7]),
        .I1(in_tdata[7]),
        .I2(data3[15]),
        .I3(\out_tdata[30]_INST_0_i_6_n_0 ),
        .I4(data_shift_next[1]),
        .I5(data_shift_next[2]),
        .O(\out_tdata[15]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFAEFE00000000)) 
    \out_tdata[15]_INST_0_i_6 
       (.I0(\out_tdata[15]_INST_0_i_7_n_0 ),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[3]),
        .I3(data_shift_next[4]),
        .I4(\out_tdata[15]_INST_0_i_8_n_0 ),
        .I5(\out_tdata[31]_INST_0_i_14_n_0 ),
        .O(\out_tdata[15]_INST_0_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT4 #(
    .INIT(16'h4FFF)) 
    \out_tdata[15]_INST_0_i_7 
       (.I0(data_shift_next[6]),
        .I1(data_shift_next[5]),
        .I2(\data_shift_reg_n_0_[7] ),
        .I3(data_shift_next[7]),
        .O(\out_tdata[15]_INST_0_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT4 #(
    .INIT(16'h4F44)) 
    \out_tdata[15]_INST_0_i_8 
       (.I0(data_shift_next[2]),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[5]),
        .I3(data_shift_next[4]),
        .O(\out_tdata[15]_INST_0_i_8_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[16]_INST_0 
       (.I0(p_1_in[16]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(in_tdata[16]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(out_tdata[16]));
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[17]_INST_0 
       (.I0(p_1_in[17]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(in_tdata[17]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(out_tdata[17]));
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[18]_INST_0 
       (.I0(p_1_in[18]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(in_tdata[18]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(out_tdata[18]));
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[19]_INST_0 
       (.I0(p_1_in[19]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(in_tdata[19]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(out_tdata[19]));
  LUT6 #(
    .INIT(64'h0000000000000700)) 
    \out_tdata[19]_INST_0_i_1 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .I2(state__0[1]),
        .I3(state__0[2]),
        .I4(state__0[0]),
        .I5(in_tctrl[3]),
        .O(\out_tdata[19]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00FF00080000FF00)) 
    \out_tdata[19]_INST_0_i_2 
       (.I0(data_shift_next[2]),
        .I1(data_shift_next[1]),
        .I2(in_tctrl[3]),
        .I3(state__0[0]),
        .I4(state__0[1]),
        .I5(state__0[2]),
        .O(\out_tdata[19]_INST_0_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[1]_INST_0 
       (.I0(p_1_in[1]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(in_tdata[1]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(out_tdata[1]));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFEA)) 
    \out_tdata[20]_INST_0 
       (.I0(\out_tdata[20]_INST_0_i_1_n_0 ),
        .I1(data3[20]),
        .I2(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I3(\out_tdata[20]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[20]_INST_0_i_3_n_0 ),
        .I5(\out_tdata[20]_INST_0_i_4_n_0 ),
        .O(out_tdata[20]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[20]_INST_0_i_1 
       (.I0(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I1(p_1_in[8]),
        .I2(\out_tdata[31]_INST_0_i_11_n_0 ),
        .I3(p_1_in[12]),
        .I4(p_1_in[28]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[20]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[20]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I1(data3[12]),
        .I2(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I3(p_1_in[20]),
        .I4(p_1_in[4]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[20]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[20]_INST_0_i_3 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[20]),
        .I2(\out_tdata[27]_INST_0_i_6_n_0 ),
        .I3(in_tdata[8]),
        .I4(in_tdata[12]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[20]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[20]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(in_tdata[4]),
        .I2(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I3(data3[16]),
        .I4(p_1_in[24]),
        .I5(\out_tdata[30]_INST_0_i_1_n_0 ),
        .O(\out_tdata[20]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFEA)) 
    \out_tdata[21]_INST_0 
       (.I0(\out_tdata[21]_INST_0_i_1_n_0 ),
        .I1(data3[21]),
        .I2(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I3(\out_tdata[21]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[21]_INST_0_i_3_n_0 ),
        .I5(\out_tdata[21]_INST_0_i_4_n_0 ),
        .O(out_tdata[21]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[21]_INST_0_i_1 
       (.I0(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I1(p_1_in[9]),
        .I2(\out_tdata[31]_INST_0_i_11_n_0 ),
        .I3(p_1_in[13]),
        .I4(p_1_in[29]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[21]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[21]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I1(data3[13]),
        .I2(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I3(p_1_in[21]),
        .I4(p_1_in[5]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[21]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[21]_INST_0_i_3 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[21]),
        .I2(\out_tdata[27]_INST_0_i_6_n_0 ),
        .I3(in_tdata[9]),
        .I4(in_tdata[13]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[21]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[21]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(in_tdata[5]),
        .I2(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I3(data3[17]),
        .I4(p_1_in[25]),
        .I5(\out_tdata[30]_INST_0_i_1_n_0 ),
        .O(\out_tdata[21]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFEA)) 
    \out_tdata[22]_INST_0 
       (.I0(\out_tdata[22]_INST_0_i_1_n_0 ),
        .I1(data3[22]),
        .I2(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I3(\out_tdata[22]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[22]_INST_0_i_3_n_0 ),
        .I5(\out_tdata[22]_INST_0_i_4_n_0 ),
        .O(out_tdata[22]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[22]_INST_0_i_1 
       (.I0(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I1(p_1_in[10]),
        .I2(\out_tdata[31]_INST_0_i_11_n_0 ),
        .I3(p_1_in[14]),
        .I4(p_1_in[30]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[22]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[22]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I1(data3[14]),
        .I2(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I3(p_1_in[22]),
        .I4(p_1_in[6]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[22]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[22]_INST_0_i_3 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[22]),
        .I2(\out_tdata[27]_INST_0_i_6_n_0 ),
        .I3(in_tdata[10]),
        .I4(in_tdata[14]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[22]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[22]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(in_tdata[6]),
        .I2(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I3(data3[18]),
        .I4(p_1_in[26]),
        .I5(\out_tdata[30]_INST_0_i_1_n_0 ),
        .O(\out_tdata[22]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFEA)) 
    \out_tdata[23]_INST_0 
       (.I0(\out_tdata[23]_INST_0_i_1_n_0 ),
        .I1(data3[23]),
        .I2(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I3(\out_tdata[23]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[23]_INST_0_i_3_n_0 ),
        .I5(\out_tdata[23]_INST_0_i_4_n_0 ),
        .O(out_tdata[23]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[23]_INST_0_i_1 
       (.I0(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I1(p_1_in[11]),
        .I2(\out_tdata[31]_INST_0_i_11_n_0 ),
        .I3(p_1_in[15]),
        .I4(p_1_in[31]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[23]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[23]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I1(data3[15]),
        .I2(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I3(p_1_in[23]),
        .I4(p_1_in[7]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[23]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[23]_INST_0_i_3 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[23]),
        .I2(\out_tdata[27]_INST_0_i_6_n_0 ),
        .I3(in_tdata[11]),
        .I4(in_tdata[15]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[23]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[23]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(in_tdata[7]),
        .I2(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I3(data3[19]),
        .I4(p_1_in[27]),
        .I5(\out_tdata[30]_INST_0_i_1_n_0 ),
        .O(\out_tdata[23]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFF8)) 
    \out_tdata[24]_INST_0 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(p_1_in[0]),
        .I2(\out_tdata[24]_INST_0_i_1_n_0 ),
        .I3(\out_tdata[24]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[24]_INST_0_i_3_n_0 ),
        .I5(\out_tdata[24]_INST_0_i_4_n_0 ),
        .O(out_tdata[24]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[24]_INST_0_i_1 
       (.I0(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I1(p_1_in[24]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(p_1_in[12]),
        .I4(p_1_in[16]),
        .I5(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[24]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[24]_INST_0_i_2 
       (.I0(\out_tdata[30]_INST_0_i_1_n_0 ),
        .I1(p_1_in[28]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[16]),
        .I4(p_1_in[8]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[24]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[24]_INST_0_i_3 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[24]),
        .I2(\out_tdata[27]_INST_0_i_6_n_0 ),
        .I3(in_tdata[12]),
        .I4(in_tdata[16]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[24]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[24]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(in_tdata[8]),
        .I2(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I3(data3[20]),
        .I4(in_tdata[0]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[24]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFF8)) 
    \out_tdata[25]_INST_0 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(p_1_in[1]),
        .I2(\out_tdata[25]_INST_0_i_1_n_0 ),
        .I3(\out_tdata[25]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[25]_INST_0_i_3_n_0 ),
        .I5(\out_tdata[25]_INST_0_i_4_n_0 ),
        .O(out_tdata[25]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[25]_INST_0_i_1 
       (.I0(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I1(p_1_in[25]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(p_1_in[13]),
        .I4(p_1_in[17]),
        .I5(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[25]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[25]_INST_0_i_2 
       (.I0(\out_tdata[30]_INST_0_i_1_n_0 ),
        .I1(p_1_in[29]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[17]),
        .I4(p_1_in[9]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[25]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[25]_INST_0_i_3 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[25]),
        .I2(\out_tdata[27]_INST_0_i_6_n_0 ),
        .I3(in_tdata[13]),
        .I4(in_tdata[17]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[25]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[25]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(in_tdata[9]),
        .I2(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I3(data3[21]),
        .I4(in_tdata[1]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[25]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFF8)) 
    \out_tdata[26]_INST_0 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(p_1_in[2]),
        .I2(\out_tdata[26]_INST_0_i_1_n_0 ),
        .I3(\out_tdata[26]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[26]_INST_0_i_3_n_0 ),
        .I5(\out_tdata[26]_INST_0_i_4_n_0 ),
        .O(out_tdata[26]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[26]_INST_0_i_1 
       (.I0(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I1(p_1_in[26]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(p_1_in[14]),
        .I4(p_1_in[18]),
        .I5(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[26]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[26]_INST_0_i_2 
       (.I0(\out_tdata[30]_INST_0_i_1_n_0 ),
        .I1(p_1_in[30]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[18]),
        .I4(p_1_in[10]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[26]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[26]_INST_0_i_3 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[26]),
        .I2(\out_tdata[27]_INST_0_i_6_n_0 ),
        .I3(in_tdata[14]),
        .I4(in_tdata[18]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[26]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[26]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(in_tdata[10]),
        .I2(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I3(data3[22]),
        .I4(in_tdata[2]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[26]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFF8)) 
    \out_tdata[27]_INST_0 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(p_1_in[3]),
        .I2(\out_tdata[27]_INST_0_i_1_n_0 ),
        .I3(\out_tdata[27]_INST_0_i_2_n_0 ),
        .I4(\out_tdata[27]_INST_0_i_3_n_0 ),
        .I5(\out_tdata[27]_INST_0_i_4_n_0 ),
        .O(out_tdata[27]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[27]_INST_0_i_1 
       (.I0(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I1(p_1_in[27]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(p_1_in[15]),
        .I4(p_1_in[19]),
        .I5(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[27]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[27]_INST_0_i_2 
       (.I0(\out_tdata[30]_INST_0_i_1_n_0 ),
        .I1(p_1_in[31]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[19]),
        .I4(p_1_in[11]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[27]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[27]_INST_0_i_3 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[27]),
        .I2(\out_tdata[27]_INST_0_i_6_n_0 ),
        .I3(in_tdata[15]),
        .I4(in_tdata[19]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[27]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[27]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(in_tdata[11]),
        .I2(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I3(data3[23]),
        .I4(in_tdata[3]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[27]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFF00000020)) 
    \out_tdata[27]_INST_0_i_5 
       (.I0(\data_shift[4]_i_3_n_0 ),
        .I1(state__0[2]),
        .I2(state__0[1]),
        .I3(state__0[0]),
        .I4(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I5(\out_tdata[59]_INST_0_i_4_n_0 ),
        .O(\out_tdata[27]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h0000000004001410)) 
    \out_tdata[27]_INST_0_i_6 
       (.I0(state__0[0]),
        .I1(state__0[2]),
        .I2(state__0[1]),
        .I3(in_tctrl[3]),
        .I4(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I5(in_tready_INST_0_i_8_n_0),
        .O(\out_tdata[27]_INST_0_i_6_n_0 ));
  LUT6 #(
    .INIT(64'h0000001000000000)) 
    \out_tdata[27]_INST_0_i_7 
       (.I0(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I1(state__0[0]),
        .I2(state__0[1]),
        .I3(state__0[2]),
        .I4(data_shift_next[1]),
        .I5(data_shift_next[2]),
        .O(\out_tdata[27]_INST_0_i_7_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFF8)) 
    \out_tdata[28]_INST_0 
       (.I0(p_1_in[31]),
        .I1(\out_tdata[30]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[30]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[28]_INST_0_i_1_n_0 ),
        .I4(\out_tdata[28]_INST_0_i_2_n_0 ),
        .I5(\out_tdata[28]_INST_0_i_3_n_0 ),
        .O(out_tdata[28]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[28]_INST_0_i_1 
       (.I0(\out_tdata[31]_INST_0_i_8_n_0 ),
        .I1(in_tdata[4]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[20]),
        .I4(p_1_in[12]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[28]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[28]_INST_0_i_2 
       (.I0(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I1(p_1_in[28]),
        .I2(\out_tdata[31]_INST_0_i_11_n_0 ),
        .I3(p_1_in[20]),
        .I4(p_1_in[4]),
        .I5(\out_tdata[31]_INST_0_i_4_n_0 ),
        .O(\out_tdata[28]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[28]_INST_0_i_3 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[28]),
        .I2(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I3(in_tdata[12]),
        .I4(in_tdata[20]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[28]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFF8)) 
    \out_tdata[29]_INST_0 
       (.I0(p_1_in[31]),
        .I1(\out_tdata[30]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[30]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[29]_INST_0_i_1_n_0 ),
        .I4(\out_tdata[29]_INST_0_i_2_n_0 ),
        .I5(\out_tdata[29]_INST_0_i_3_n_0 ),
        .O(out_tdata[29]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[29]_INST_0_i_1 
       (.I0(\out_tdata[31]_INST_0_i_8_n_0 ),
        .I1(in_tdata[5]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[21]),
        .I4(p_1_in[13]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[29]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[29]_INST_0_i_2 
       (.I0(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I1(p_1_in[29]),
        .I2(\out_tdata[31]_INST_0_i_11_n_0 ),
        .I3(p_1_in[21]),
        .I4(p_1_in[5]),
        .I5(\out_tdata[31]_INST_0_i_4_n_0 ),
        .O(\out_tdata[29]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[29]_INST_0_i_3 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[29]),
        .I2(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I3(in_tdata[13]),
        .I4(in_tdata[21]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[29]_INST_0_i_3_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[2]_INST_0 
       (.I0(p_1_in[2]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(in_tdata[2]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(out_tdata[2]));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFF8)) 
    \out_tdata[30]_INST_0 
       (.I0(p_1_in[31]),
        .I1(\out_tdata[30]_INST_0_i_1_n_0 ),
        .I2(\out_tdata[30]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[30]_INST_0_i_3_n_0 ),
        .I4(\out_tdata[30]_INST_0_i_4_n_0 ),
        .I5(\out_tdata[30]_INST_0_i_5_n_0 ),
        .O(out_tdata[30]));
  LUT6 #(
    .INIT(64'h0400040004040400)) 
    \out_tdata[30]_INST_0_i_1 
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I4(data_shift_next[1]),
        .I5(data_shift_next[2]),
        .O(\out_tdata[30]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hCC00F0000000AA00)) 
    \out_tdata[30]_INST_0_i_2 
       (.I0(p_1_in[15]),
        .I1(in_tdata[15]),
        .I2(data3[23]),
        .I3(\out_tdata[30]_INST_0_i_6_n_0 ),
        .I4(data_shift_next[1]),
        .I5(data_shift_next[2]),
        .O(\out_tdata[30]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[30]_INST_0_i_3 
       (.I0(\out_tdata[31]_INST_0_i_8_n_0 ),
        .I1(in_tdata[6]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[22]),
        .I4(p_1_in[14]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[30]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[30]_INST_0_i_4 
       (.I0(\out_tdata[30]_INST_0_i_7_n_0 ),
        .I1(p_1_in[30]),
        .I2(\out_tdata[31]_INST_0_i_11_n_0 ),
        .I3(p_1_in[22]),
        .I4(p_1_in[6]),
        .I5(\out_tdata[31]_INST_0_i_4_n_0 ),
        .O(\out_tdata[30]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[30]_INST_0_i_5 
       (.I0(\out_tdata[19]_INST_0_i_2_n_0 ),
        .I1(in_tdata[30]),
        .I2(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I3(in_tdata[14]),
        .I4(in_tdata[22]),
        .I5(\out_tdata[31]_INST_0_i_6_n_0 ),
        .O(\out_tdata[30]_INST_0_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'h04000000)) 
    \out_tdata[30]_INST_0_i_6 
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .I3(data_shift_next[4]),
        .I4(data_shift_next[3]),
        .O(\out_tdata[30]_INST_0_i_6_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \out_tdata[30]_INST_0_i_7 
       (.I0(\out_tdata[15]_INST_0_i_6_n_0 ),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .O(\out_tdata[30]_INST_0_i_7_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFEFEFE)) 
    \out_tdata[31]_INST_0 
       (.I0(\out_tdata[31]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[31]_INST_0_i_2_n_0 ),
        .I2(\out_tdata[31]_INST_0_i_3_n_0 ),
        .I3(p_1_in[7]),
        .I4(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I5(\out_tdata[31]_INST_0_i_5_n_0 ),
        .O(out_tdata[31]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[31]_INST_0_i_1 
       (.I0(\out_tdata[31]_INST_0_i_6_n_0 ),
        .I1(in_tdata[23]),
        .I2(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I3(in_tdata[15]),
        .I4(in_tdata[7]),
        .I5(\out_tdata[31]_INST_0_i_8_n_0 ),
        .O(\out_tdata[31]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT3 #(
    .INIT(8'h20)) 
    \out_tdata[31]_INST_0_i_10 
       (.I0(\out_tdata[31]_INST_0_i_17_n_0 ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .O(\out_tdata[31]_INST_0_i_10_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFF00000002)) 
    \out_tdata[31]_INST_0_i_11 
       (.I0(\out_tdata[31]_INST_0_i_18_n_0 ),
        .I1(data_shift_next[3]),
        .I2(data_shift_next[1]),
        .I3(data_shift_next[2]),
        .I4(data_shift_next[4]),
        .I5(\out_tdata[63]_INST_0_i_2_n_0 ),
        .O(\out_tdata[31]_INST_0_i_11_n_0 ));
  LUT6 #(
    .INIT(64'h0888888808880800)) 
    \out_tdata[31]_INST_0_i_12 
       (.I0(p_1_in[31]),
        .I1(\out_tdata[31]_INST_0_i_19_n_0 ),
        .I2(data_shift_next[4]),
        .I3(data_shift_next[3]),
        .I4(data_shift_next[2]),
        .I5(data_shift_next[1]),
        .O(\out_tdata[31]_INST_0_i_12_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'h00000001)) 
    \out_tdata[31]_INST_0_i_13 
       (.I0(data_shift_next[4]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(data_shift_next[3]),
        .I4(data_shift_next[5]),
        .O(\out_tdata[31]_INST_0_i_13_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT3 #(
    .INIT(8'h08)) 
    \out_tdata[31]_INST_0_i_14 
       (.I0(state__0[1]),
        .I1(state__0[0]),
        .I2(state__0[2]),
        .O(\out_tdata[31]_INST_0_i_14_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT5 #(
    .INIT(32'h02000F00)) 
    \out_tdata[31]_INST_0_i_15 
       (.I0(data_shift_next[4]),
        .I1(data_shift_next[5]),
        .I2(state__0[2]),
        .I3(state__0[1]),
        .I4(state__0[0]),
        .O(\out_tdata[31]_INST_0_i_15_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT5 #(
    .INIT(32'h3FBFFFBF)) 
    \out_tdata[31]_INST_0_i_16 
       (.I0(data_shift_next[3]),
        .I1(data_shift_next[7]),
        .I2(\data_shift_reg_n_0_[7] ),
        .I3(data_shift_next[5]),
        .I4(data_shift_next[6]),
        .O(\out_tdata[31]_INST_0_i_16_n_0 ));
  LUT6 #(
    .INIT(64'h0000000080000000)) 
    \out_tdata[31]_INST_0_i_17 
       (.I0(data_shift_next[5]),
        .I1(\out_tdata[31]_INST_0_i_14_n_0 ),
        .I2(data_shift_next[6]),
        .I3(data_shift_next[7]),
        .I4(\data_shift_reg_n_0_[7] ),
        .I5(\out_tdata[63]_INST_0_i_7_n_0 ),
        .O(\out_tdata[31]_INST_0_i_17_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT5 #(
    .INIT(32'h80000000)) 
    \out_tdata[31]_INST_0_i_18 
       (.I0(\data_shift_reg_n_0_[7] ),
        .I1(data_shift_next[7]),
        .I2(data_shift_next[6]),
        .I3(\out_tdata[31]_INST_0_i_14_n_0 ),
        .I4(data_shift_next[5]),
        .O(\out_tdata[31]_INST_0_i_18_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \out_tdata[31]_INST_0_i_19 
       (.I0(state__0[1]),
        .I1(state__0[2]),
        .O(\out_tdata[31]_INST_0_i_19_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[31]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I1(data3[23]),
        .I2(\out_tdata[31]_INST_0_i_10_n_0 ),
        .I3(p_1_in[15]),
        .I4(p_1_in[23]),
        .I5(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[31]_INST_0_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFEEE)) 
    \out_tdata[31]_INST_0_i_3 
       (.I0(\out_tdata[30]_INST_0_i_2_n_0 ),
        .I1(\out_tdata[31]_INST_0_i_12_n_0 ),
        .I2(in_tdata[31]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(\out_tdata[31]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFF20000000)) 
    \out_tdata[31]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_13_n_0 ),
        .I1(data_shift_next[6]),
        .I2(\out_tdata[31]_INST_0_i_14_n_0 ),
        .I3(\data_shift_reg_n_0_[7] ),
        .I4(data_shift_next[7]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(\out_tdata[31]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFF0000F4440000)) 
    \out_tdata[31]_INST_0_i_5 
       (.I0(data_shift_next[3]),
        .I1(\out_tdata[31]_INST_0_i_15_n_0 ),
        .I2(\out_tdata[31]_INST_0_i_14_n_0 ),
        .I3(\out_tdata[31]_INST_0_i_16_n_0 ),
        .I4(p_1_in[31]),
        .I5(\out_tdata[19]_INST_0_i_1_n_0 ),
        .O(\out_tdata[31]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h00000000FFFF0008)) 
    \out_tdata[31]_INST_0_i_6 
       (.I0(state__0[0]),
        .I1(state__0[2]),
        .I2(state__0[1]),
        .I3(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I4(\out_tdata[31]_INST_0_i_17_n_0 ),
        .I5(in_tready_INST_0_i_8_n_0),
        .O(\out_tdata[31]_INST_0_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'h80000000)) 
    \out_tdata[31]_INST_0_i_7 
       (.I0(\out_tdata[31]_INST_0_i_13_n_0 ),
        .I1(\data_shift_reg_n_0_[7] ),
        .I2(data_shift_next[7]),
        .I3(data_shift_next[6]),
        .I4(\out_tdata[31]_INST_0_i_14_n_0 ),
        .O(\out_tdata[31]_INST_0_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT4 #(
    .INIT(16'hABAA)) 
    \out_tdata[31]_INST_0_i_8 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[2]),
        .I3(\out_tdata[31]_INST_0_i_17_n_0 ),
        .O(\out_tdata[31]_INST_0_i_8_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT5 #(
    .INIT(32'h00020000)) 
    \out_tdata[31]_INST_0_i_9 
       (.I0(data_shift_next[4]),
        .I1(data_shift_next[2]),
        .I2(data_shift_next[1]),
        .I3(data_shift_next[3]),
        .I4(\out_tdata[31]_INST_0_i_18_n_0 ),
        .O(\out_tdata[31]_INST_0_i_9_n_0 ));
  LUT4 #(
    .INIT(16'h4200)) 
    \out_tdata[32]_INST_0 
       (.I0(state__0[0]),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(in_tdata[32]),
        .O(out_tdata[32]));
  LUT4 #(
    .INIT(16'h4200)) 
    \out_tdata[33]_INST_0 
       (.I0(state__0[0]),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(in_tdata[33]),
        .O(out_tdata[33]));
  LUT4 #(
    .INIT(16'h4200)) 
    \out_tdata[34]_INST_0 
       (.I0(state__0[0]),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(in_tdata[34]),
        .O(out_tdata[34]));
  LUT4 #(
    .INIT(16'h4200)) 
    \out_tdata[35]_INST_0 
       (.I0(state__0[0]),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(in_tdata[35]),
        .O(out_tdata[35]));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[36]_INST_0 
       (.I0(\out_tdata[36]_INST_0_i_1_n_0 ),
        .I1(in_tdata[16]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[36]_INST_0_i_2_n_0 ),
        .I4(p_1_in[8]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[36]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[36]_INST_0_i_1 
       (.I0(in_tdata[24]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[36]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[36]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[36]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[8]),
        .I2(\out_tdata[59]_INST_0_i_4_n_0 ),
        .I3(p_1_in[16]),
        .I4(p_1_in[24]),
        .I5(\out_tdata[63]_INST_0_i_2_n_0 ),
        .O(\out_tdata[36]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[37]_INST_0 
       (.I0(\out_tdata[37]_INST_0_i_1_n_0 ),
        .I1(in_tdata[17]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[37]_INST_0_i_2_n_0 ),
        .I4(p_1_in[9]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[37]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[37]_INST_0_i_1 
       (.I0(in_tdata[25]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[37]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[37]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[37]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[9]),
        .I2(\out_tdata[59]_INST_0_i_4_n_0 ),
        .I3(p_1_in[17]),
        .I4(p_1_in[25]),
        .I5(\out_tdata[63]_INST_0_i_2_n_0 ),
        .O(\out_tdata[37]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[38]_INST_0 
       (.I0(\out_tdata[38]_INST_0_i_1_n_0 ),
        .I1(in_tdata[18]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[38]_INST_0_i_2_n_0 ),
        .I4(p_1_in[10]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[38]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[38]_INST_0_i_1 
       (.I0(in_tdata[26]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[38]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[38]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[38]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[10]),
        .I2(\out_tdata[59]_INST_0_i_4_n_0 ),
        .I3(p_1_in[18]),
        .I4(p_1_in[26]),
        .I5(\out_tdata[63]_INST_0_i_2_n_0 ),
        .O(\out_tdata[38]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[39]_INST_0 
       (.I0(\out_tdata[39]_INST_0_i_1_n_0 ),
        .I1(in_tdata[19]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[39]_INST_0_i_2_n_0 ),
        .I4(p_1_in[11]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[39]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[39]_INST_0_i_1 
       (.I0(in_tdata[27]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[39]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[39]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[39]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[11]),
        .I2(\out_tdata[59]_INST_0_i_4_n_0 ),
        .I3(p_1_in[19]),
        .I4(p_1_in[27]),
        .I5(\out_tdata[63]_INST_0_i_2_n_0 ),
        .O(\out_tdata[39]_INST_0_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[3]_INST_0 
       (.I0(p_1_in[3]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(in_tdata[3]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(out_tdata[3]));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[40]_INST_0 
       (.I0(\out_tdata[40]_INST_0_i_1_n_0 ),
        .I1(in_tdata[20]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[40]_INST_0_i_2_n_0 ),
        .I4(p_1_in[12]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[40]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[40]_INST_0_i_1 
       (.I0(in_tdata[28]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[40]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[40]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[40]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[12]),
        .I2(\out_tdata[59]_INST_0_i_4_n_0 ),
        .I3(p_1_in[20]),
        .I4(p_1_in[28]),
        .I5(\out_tdata[63]_INST_0_i_2_n_0 ),
        .O(\out_tdata[40]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[41]_INST_0 
       (.I0(\out_tdata[41]_INST_0_i_1_n_0 ),
        .I1(in_tdata[21]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[41]_INST_0_i_2_n_0 ),
        .I4(p_1_in[13]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[41]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[41]_INST_0_i_1 
       (.I0(in_tdata[29]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[41]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[41]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[41]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[13]),
        .I2(\out_tdata[59]_INST_0_i_4_n_0 ),
        .I3(p_1_in[21]),
        .I4(p_1_in[29]),
        .I5(\out_tdata[63]_INST_0_i_2_n_0 ),
        .O(\out_tdata[41]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[42]_INST_0 
       (.I0(\out_tdata[42]_INST_0_i_1_n_0 ),
        .I1(in_tdata[22]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[42]_INST_0_i_2_n_0 ),
        .I4(p_1_in[14]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[42]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[42]_INST_0_i_1 
       (.I0(in_tdata[30]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[42]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[42]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[42]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[14]),
        .I2(\out_tdata[59]_INST_0_i_4_n_0 ),
        .I3(p_1_in[22]),
        .I4(p_1_in[30]),
        .I5(\out_tdata[63]_INST_0_i_2_n_0 ),
        .O(\out_tdata[42]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[43]_INST_0 
       (.I0(\out_tdata[43]_INST_0_i_1_n_0 ),
        .I1(in_tdata[23]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[43]_INST_0_i_2_n_0 ),
        .I4(p_1_in[15]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[43]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[43]_INST_0_i_1 
       (.I0(in_tdata[31]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[43]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[43]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[43]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[15]),
        .I2(\out_tdata[59]_INST_0_i_4_n_0 ),
        .I3(p_1_in[23]),
        .I4(p_1_in[31]),
        .I5(\out_tdata[63]_INST_0_i_2_n_0 ),
        .O(\out_tdata[43]_INST_0_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAEAEA)) 
    \out_tdata[44]_INST_0 
       (.I0(\out_tdata[44]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I2(in_tdata[0]),
        .I3(p_1_in[16]),
        .I4(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[44]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[44]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[44]),
        .I2(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I3(in_tdata[32]),
        .I4(in_tdata[16]),
        .I5(\out_tdata[63]_INST_0_i_6_n_0 ),
        .O(\out_tdata[44]_INST_0_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAEAEA)) 
    \out_tdata[45]_INST_0 
       (.I0(\out_tdata[45]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I2(in_tdata[1]),
        .I3(p_1_in[17]),
        .I4(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[45]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[45]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[45]),
        .I2(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I3(in_tdata[33]),
        .I4(in_tdata[17]),
        .I5(\out_tdata[63]_INST_0_i_6_n_0 ),
        .O(\out_tdata[45]_INST_0_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAEAEA)) 
    \out_tdata[46]_INST_0 
       (.I0(\out_tdata[46]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I2(in_tdata[2]),
        .I3(p_1_in[18]),
        .I4(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[46]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[46]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[46]),
        .I2(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I3(in_tdata[34]),
        .I4(in_tdata[18]),
        .I5(\out_tdata[63]_INST_0_i_6_n_0 ),
        .O(\out_tdata[46]_INST_0_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAEAEA)) 
    \out_tdata[47]_INST_0 
       (.I0(\out_tdata[47]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I2(in_tdata[3]),
        .I3(p_1_in[19]),
        .I4(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[47]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[47]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[47]),
        .I2(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I3(in_tdata[35]),
        .I4(in_tdata[19]),
        .I5(\out_tdata[63]_INST_0_i_6_n_0 ),
        .O(\out_tdata[47]_INST_0_i_1_n_0 ));
  LUT4 #(
    .INIT(16'h4200)) 
    \out_tdata[48]_INST_0 
       (.I0(state__0[0]),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(in_tdata[48]),
        .O(out_tdata[48]));
  LUT4 #(
    .INIT(16'h4200)) 
    \out_tdata[49]_INST_0 
       (.I0(state__0[0]),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(in_tdata[49]),
        .O(out_tdata[49]));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFEFEFE)) 
    \out_tdata[4]_INST_0 
       (.I0(\out_tdata[4]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[4]_INST_0_i_2_n_0 ),
        .I2(\out_tdata[4]_INST_0_i_3_n_0 ),
        .I3(p_1_in[16]),
        .I4(\out_tdata[11]_INST_0_i_4_n_0 ),
        .I5(\out_tdata[4]_INST_0_i_4_n_0 ),
        .O(out_tdata[4]));
  LUT4 #(
    .INIT(16'hEEEA)) 
    \out_tdata[4]_INST_0_i_1 
       (.I0(\out_tdata[4]_INST_0_i_5_n_0 ),
        .I1(p_1_in[0]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[4]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT3 #(
    .INIT(8'hE0)) 
    \out_tdata[4]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I2(data3[8]),
        .O(\out_tdata[4]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[4]_INST_0_i_3 
       (.I0(in_tdata[0]),
        .I1(\out_tdata[11]_INST_0_i_7_n_0 ),
        .I2(in_tdata[4]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(\out_tdata[4]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[4]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(p_1_in[24]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[0]),
        .I4(data3[16]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[4]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[4]_INST_0_i_5 
       (.I0(p_1_in[4]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(p_1_in[8]),
        .I3(\out_tdata[15]_INST_0_i_6_n_0 ),
        .O(\out_tdata[4]_INST_0_i_5_n_0 ));
  LUT4 #(
    .INIT(16'h4200)) 
    \out_tdata[50]_INST_0 
       (.I0(state__0[0]),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(in_tdata[50]),
        .O(out_tdata[50]));
  LUT4 #(
    .INIT(16'h4200)) 
    \out_tdata[51]_INST_0 
       (.I0(state__0[0]),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(in_tdata[51]),
        .O(out_tdata[51]));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[52]_INST_0 
       (.I0(\out_tdata[52]_INST_0_i_1_n_0 ),
        .I1(in_tdata[24]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[52]_INST_0_i_2_n_0 ),
        .I4(p_1_in[20]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[52]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[52]_INST_0_i_1 
       (.I0(in_tdata[36]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[52]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[52]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[52]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[20]),
        .I2(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I3(in_tdata[4]),
        .I4(p_1_in[24]),
        .I5(\out_tdata[59]_INST_0_i_4_n_0 ),
        .O(\out_tdata[52]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[53]_INST_0 
       (.I0(\out_tdata[53]_INST_0_i_1_n_0 ),
        .I1(in_tdata[25]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[53]_INST_0_i_2_n_0 ),
        .I4(p_1_in[21]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[53]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[53]_INST_0_i_1 
       (.I0(in_tdata[37]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[53]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[53]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[53]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[21]),
        .I2(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I3(in_tdata[5]),
        .I4(p_1_in[25]),
        .I5(\out_tdata[59]_INST_0_i_4_n_0 ),
        .O(\out_tdata[53]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[54]_INST_0 
       (.I0(\out_tdata[54]_INST_0_i_1_n_0 ),
        .I1(in_tdata[26]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[54]_INST_0_i_2_n_0 ),
        .I4(p_1_in[22]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[54]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[54]_INST_0_i_1 
       (.I0(in_tdata[38]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[54]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[54]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[54]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[22]),
        .I2(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I3(in_tdata[6]),
        .I4(p_1_in[26]),
        .I5(\out_tdata[59]_INST_0_i_4_n_0 ),
        .O(\out_tdata[54]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[55]_INST_0 
       (.I0(\out_tdata[55]_INST_0_i_1_n_0 ),
        .I1(in_tdata[27]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[55]_INST_0_i_2_n_0 ),
        .I4(p_1_in[23]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[55]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[55]_INST_0_i_1 
       (.I0(in_tdata[39]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[55]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[55]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[55]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[23]),
        .I2(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I3(in_tdata[7]),
        .I4(p_1_in[27]),
        .I5(\out_tdata[59]_INST_0_i_4_n_0 ),
        .O(\out_tdata[55]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[56]_INST_0 
       (.I0(\out_tdata[56]_INST_0_i_1_n_0 ),
        .I1(in_tdata[28]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[56]_INST_0_i_2_n_0 ),
        .I4(p_1_in[24]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[56]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[56]_INST_0_i_1 
       (.I0(in_tdata[40]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[56]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[56]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[56]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[24]),
        .I2(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I3(in_tdata[8]),
        .I4(p_1_in[28]),
        .I5(\out_tdata[59]_INST_0_i_4_n_0 ),
        .O(\out_tdata[56]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[57]_INST_0 
       (.I0(\out_tdata[57]_INST_0_i_1_n_0 ),
        .I1(in_tdata[29]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[57]_INST_0_i_2_n_0 ),
        .I4(p_1_in[25]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[57]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[57]_INST_0_i_1 
       (.I0(in_tdata[41]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[57]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[57]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[57]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[25]),
        .I2(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I3(in_tdata[9]),
        .I4(p_1_in[29]),
        .I5(\out_tdata[59]_INST_0_i_4_n_0 ),
        .O(\out_tdata[57]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[58]_INST_0 
       (.I0(\out_tdata[58]_INST_0_i_1_n_0 ),
        .I1(in_tdata[30]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[58]_INST_0_i_2_n_0 ),
        .I4(p_1_in[26]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[58]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[58]_INST_0_i_1 
       (.I0(in_tdata[42]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[58]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[58]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[58]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[26]),
        .I2(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I3(in_tdata[10]),
        .I4(p_1_in[30]),
        .I5(\out_tdata[59]_INST_0_i_4_n_0 ),
        .O(\out_tdata[58]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFEAFFEAFFEA)) 
    \out_tdata[59]_INST_0 
       (.I0(\out_tdata[59]_INST_0_i_1_n_0 ),
        .I1(in_tdata[31]),
        .I2(\out_tdata[59]_INST_0_i_2_n_0 ),
        .I3(\out_tdata[59]_INST_0_i_3_n_0 ),
        .I4(p_1_in[27]),
        .I5(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[59]));
  LUT6 #(
    .INIT(64'h888888F8F8888888)) 
    \out_tdata[59]_INST_0_i_1 
       (.I0(in_tdata[43]),
        .I1(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I2(in_tdata[59]),
        .I3(state__0[2]),
        .I4(state__0[1]),
        .I5(state__0[0]),
        .O(\out_tdata[59]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000080000000000)) 
    \out_tdata[59]_INST_0_i_2 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .I2(state__0[1]),
        .I3(state__0[2]),
        .I4(state__0[0]),
        .I5(in_tctrl[3]),
        .O(\out_tdata[59]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[59]_INST_0_i_3 
       (.I0(\out_tdata[63]_INST_0_i_6_n_0 ),
        .I1(in_tdata[27]),
        .I2(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I3(in_tdata[11]),
        .I4(p_1_in[31]),
        .I5(\out_tdata[59]_INST_0_i_4_n_0 ),
        .O(\out_tdata[59]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h0000070000000000)) 
    \out_tdata[59]_INST_0_i_4 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .I2(state__0[1]),
        .I3(state__0[2]),
        .I4(state__0[0]),
        .I5(in_tctrl[3]),
        .O(\out_tdata[59]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFEFEFE)) 
    \out_tdata[5]_INST_0 
       (.I0(\out_tdata[5]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[5]_INST_0_i_2_n_0 ),
        .I2(\out_tdata[5]_INST_0_i_3_n_0 ),
        .I3(p_1_in[17]),
        .I4(\out_tdata[11]_INST_0_i_4_n_0 ),
        .I5(\out_tdata[5]_INST_0_i_4_n_0 ),
        .O(out_tdata[5]));
  LUT4 #(
    .INIT(16'hEEEA)) 
    \out_tdata[5]_INST_0_i_1 
       (.I0(\out_tdata[5]_INST_0_i_5_n_0 ),
        .I1(p_1_in[1]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[5]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT3 #(
    .INIT(8'hE0)) 
    \out_tdata[5]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I2(data3[9]),
        .O(\out_tdata[5]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[5]_INST_0_i_3 
       (.I0(in_tdata[1]),
        .I1(\out_tdata[11]_INST_0_i_7_n_0 ),
        .I2(in_tdata[5]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(\out_tdata[5]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[5]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(p_1_in[25]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[1]),
        .I4(data3[17]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[5]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[5]_INST_0_i_5 
       (.I0(p_1_in[5]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(p_1_in[9]),
        .I3(\out_tdata[15]_INST_0_i_6_n_0 ),
        .O(\out_tdata[5]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAEAEA)) 
    \out_tdata[60]_INST_0 
       (.I0(\out_tdata[60]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I2(in_tdata[12]),
        .I3(p_1_in[28]),
        .I4(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[60]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[60]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[60]),
        .I2(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I3(in_tdata[44]),
        .I4(in_tdata[28]),
        .I5(\out_tdata[63]_INST_0_i_6_n_0 ),
        .O(\out_tdata[60]_INST_0_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAEAEA)) 
    \out_tdata[61]_INST_0 
       (.I0(\out_tdata[61]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I2(in_tdata[13]),
        .I3(p_1_in[29]),
        .I4(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[61]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[61]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[61]),
        .I2(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I3(in_tdata[45]),
        .I4(in_tdata[29]),
        .I5(\out_tdata[63]_INST_0_i_6_n_0 ),
        .O(\out_tdata[61]_INST_0_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAEAEA)) 
    \out_tdata[62]_INST_0 
       (.I0(\out_tdata[62]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I2(in_tdata[14]),
        .I3(p_1_in[30]),
        .I4(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[62]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[62]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[62]),
        .I2(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I3(in_tdata[46]),
        .I4(in_tdata[30]),
        .I5(\out_tdata[63]_INST_0_i_6_n_0 ),
        .O(\out_tdata[62]_INST_0_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFEAEAEA)) 
    \out_tdata[63]_INST_0 
       (.I0(\out_tdata[63]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[63]_INST_0_i_2_n_0 ),
        .I2(in_tdata[15]),
        .I3(p_1_in[31]),
        .I4(\out_tdata[63]_INST_0_i_3_n_0 ),
        .O(out_tdata[63]));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[63]_INST_0_i_1 
       (.I0(\out_tdata[63]_INST_0_i_4_n_0 ),
        .I1(in_tdata[63]),
        .I2(\out_tdata[63]_INST_0_i_5_n_0 ),
        .I3(in_tdata[47]),
        .I4(in_tdata[31]),
        .I5(\out_tdata[63]_INST_0_i_6_n_0 ),
        .O(\out_tdata[63]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000001000)) 
    \out_tdata[63]_INST_0_i_2 
       (.I0(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(state__0[0]),
        .I4(data_shift_next[2]),
        .I5(data_shift_next[1]),
        .O(\out_tdata[63]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0800080008080800)) 
    \out_tdata[63]_INST_0_i_3 
       (.I0(state__0[0]),
        .I1(state__0[2]),
        .I2(state__0[1]),
        .I3(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I4(data_shift_next[1]),
        .I5(data_shift_next[2]),
        .O(\out_tdata[63]_INST_0_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT3 #(
    .INIT(8'h18)) 
    \out_tdata[63]_INST_0_i_4 
       (.I0(state__0[2]),
        .I1(state__0[1]),
        .I2(state__0[0]),
        .O(\out_tdata[63]_INST_0_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h0000000008000000)) 
    \out_tdata[63]_INST_0_i_5 
       (.I0(data_shift_next[3]),
        .I1(data_shift_next[4]),
        .I2(state__0[1]),
        .I3(state__0[2]),
        .I4(state__0[0]),
        .I5(in_tready_INST_0_i_8_n_0),
        .O(\out_tdata[63]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h0004000000000000)) 
    \out_tdata[63]_INST_0_i_6 
       (.I0(data_shift_next[1]),
        .I1(data_shift_next[2]),
        .I2(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I3(state__0[1]),
        .I4(state__0[2]),
        .I5(state__0[0]),
        .O(\out_tdata[63]_INST_0_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT2 #(
    .INIT(4'h7)) 
    \out_tdata[63]_INST_0_i_7 
       (.I0(data_shift_next[3]),
        .I1(data_shift_next[4]),
        .O(\out_tdata[63]_INST_0_i_7_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFEFEFE)) 
    \out_tdata[6]_INST_0 
       (.I0(\out_tdata[6]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[6]_INST_0_i_2_n_0 ),
        .I2(\out_tdata[6]_INST_0_i_3_n_0 ),
        .I3(p_1_in[18]),
        .I4(\out_tdata[11]_INST_0_i_4_n_0 ),
        .I5(\out_tdata[6]_INST_0_i_4_n_0 ),
        .O(out_tdata[6]));
  LUT4 #(
    .INIT(16'hEEEA)) 
    \out_tdata[6]_INST_0_i_1 
       (.I0(\out_tdata[6]_INST_0_i_5_n_0 ),
        .I1(p_1_in[2]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[6]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT3 #(
    .INIT(8'hE0)) 
    \out_tdata[6]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I2(data3[10]),
        .O(\out_tdata[6]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[6]_INST_0_i_3 
       (.I0(in_tdata[2]),
        .I1(\out_tdata[11]_INST_0_i_7_n_0 ),
        .I2(in_tdata[6]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(\out_tdata[6]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[6]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(p_1_in[26]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[2]),
        .I4(data3[18]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[6]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[6]_INST_0_i_5 
       (.I0(p_1_in[6]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(p_1_in[10]),
        .I3(\out_tdata[15]_INST_0_i_6_n_0 ),
        .O(\out_tdata[6]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFEFEFE)) 
    \out_tdata[7]_INST_0 
       (.I0(\out_tdata[7]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[7]_INST_0_i_2_n_0 ),
        .I2(\out_tdata[7]_INST_0_i_3_n_0 ),
        .I3(p_1_in[19]),
        .I4(\out_tdata[11]_INST_0_i_4_n_0 ),
        .I5(\out_tdata[7]_INST_0_i_4_n_0 ),
        .O(out_tdata[7]));
  LUT4 #(
    .INIT(16'hEEEA)) 
    \out_tdata[7]_INST_0_i_1 
       (.I0(\out_tdata[7]_INST_0_i_5_n_0 ),
        .I1(p_1_in[3]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[7]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT3 #(
    .INIT(8'hE0)) 
    \out_tdata[7]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I2(data3[11]),
        .O(\out_tdata[7]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[7]_INST_0_i_3 
       (.I0(in_tdata[3]),
        .I1(\out_tdata[11]_INST_0_i_7_n_0 ),
        .I2(in_tdata[7]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(\out_tdata[7]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[7]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(p_1_in[27]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[3]),
        .I4(data3[19]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[7]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[7]_INST_0_i_5 
       (.I0(p_1_in[7]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(p_1_in[11]),
        .I3(\out_tdata[15]_INST_0_i_6_n_0 ),
        .O(\out_tdata[7]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFEFEFE)) 
    \out_tdata[8]_INST_0 
       (.I0(\out_tdata[8]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[8]_INST_0_i_2_n_0 ),
        .I2(\out_tdata[8]_INST_0_i_3_n_0 ),
        .I3(p_1_in[20]),
        .I4(\out_tdata[11]_INST_0_i_4_n_0 ),
        .I5(\out_tdata[8]_INST_0_i_4_n_0 ),
        .O(out_tdata[8]));
  LUT4 #(
    .INIT(16'hEEEA)) 
    \out_tdata[8]_INST_0_i_1 
       (.I0(\out_tdata[8]_INST_0_i_5_n_0 ),
        .I1(p_1_in[4]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[8]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT3 #(
    .INIT(8'hE0)) 
    \out_tdata[8]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I2(data3[12]),
        .O(\out_tdata[8]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[8]_INST_0_i_3 
       (.I0(in_tdata[4]),
        .I1(\out_tdata[11]_INST_0_i_7_n_0 ),
        .I2(in_tdata[8]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(\out_tdata[8]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[8]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(p_1_in[28]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[4]),
        .I4(data3[20]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[8]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[8]_INST_0_i_5 
       (.I0(p_1_in[8]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(p_1_in[12]),
        .I3(\out_tdata[15]_INST_0_i_6_n_0 ),
        .O(\out_tdata[8]_INST_0_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFEFEFE)) 
    \out_tdata[9]_INST_0 
       (.I0(\out_tdata[9]_INST_0_i_1_n_0 ),
        .I1(\out_tdata[9]_INST_0_i_2_n_0 ),
        .I2(\out_tdata[9]_INST_0_i_3_n_0 ),
        .I3(p_1_in[21]),
        .I4(\out_tdata[11]_INST_0_i_4_n_0 ),
        .I5(\out_tdata[9]_INST_0_i_4_n_0 ),
        .O(out_tdata[9]));
  LUT4 #(
    .INIT(16'hEEEA)) 
    \out_tdata[9]_INST_0_i_1 
       (.I0(\out_tdata[9]_INST_0_i_5_n_0 ),
        .I1(p_1_in[5]),
        .I2(\out_tdata[27]_INST_0_i_5_n_0 ),
        .I3(\out_tdata[31]_INST_0_i_11_n_0 ),
        .O(\out_tdata[9]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT3 #(
    .INIT(8'hE0)) 
    \out_tdata[9]_INST_0_i_2 
       (.I0(\out_tdata[31]_INST_0_i_4_n_0 ),
        .I1(\out_tdata[27]_INST_0_i_7_n_0 ),
        .I2(data3[13]),
        .O(\out_tdata[9]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[9]_INST_0_i_3 
       (.I0(in_tdata[5]),
        .I1(\out_tdata[11]_INST_0_i_7_n_0 ),
        .I2(in_tdata[9]),
        .I3(\out_tdata[19]_INST_0_i_2_n_0 ),
        .O(\out_tdata[9]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFF888F888F888)) 
    \out_tdata[9]_INST_0_i_4 
       (.I0(\out_tdata[31]_INST_0_i_7_n_0 ),
        .I1(p_1_in[29]),
        .I2(\out_tdata[31]_INST_0_i_9_n_0 ),
        .I3(data3[5]),
        .I4(data3[21]),
        .I5(\out_tdata[31]_INST_0_i_10_n_0 ),
        .O(\out_tdata[9]_INST_0_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \out_tdata[9]_INST_0_i_5 
       (.I0(p_1_in[9]),
        .I1(\out_tdata[19]_INST_0_i_1_n_0 ),
        .I2(p_1_in[13]),
        .I3(\out_tdata[15]_INST_0_i_6_n_0 ),
        .O(\out_tdata[9]_INST_0_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h0008FFFB)) 
    out_tlast_INST_0
       (.I0(in_tlast),
        .I1(state__0[0]),
        .I2(state__0[2]),
        .I3(state__0[1]),
        .I4(out_tlast_INST_0_i_1_n_0),
        .O(out_tlast));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT4 #(
    .INIT(16'hFFFE)) 
    out_tlast_INST_0_i_1
       (.I0(out_tlast_INST_0_i_2_n_0),
        .I1(out_tlast_INST_0_i_3_n_0),
        .I2(out_tlast_INST_0_i_4_n_0),
        .I3(out_tlast_INST_0_i_5_n_0),
        .O(out_tlast_INST_0_i_1_n_0));
  LUT4 #(
    .INIT(16'hFFFE)) 
    out_tlast_INST_0_i_2
       (.I0(xfer_count_reg[5]),
        .I1(xfer_count_reg[4]),
        .I2(xfer_count_reg[7]),
        .I3(xfer_count_reg[6]),
        .O(out_tlast_INST_0_i_2_n_0));
  LUT4 #(
    .INIT(16'hFFFE)) 
    out_tlast_INST_0_i_3
       (.I0(xfer_count_reg[1]),
        .I1(xfer_count_reg[0]),
        .I2(xfer_count_reg[3]),
        .I3(xfer_count_reg[2]),
        .O(out_tlast_INST_0_i_3_n_0));
  LUT4 #(
    .INIT(16'hFFFE)) 
    out_tlast_INST_0_i_4
       (.I0(xfer_count_reg[13]),
        .I1(xfer_count_reg[12]),
        .I2(xfer_count_reg[15]),
        .I3(xfer_count_reg[14]),
        .O(out_tlast_INST_0_i_4_n_0));
  LUT4 #(
    .INIT(16'hFFFE)) 
    out_tlast_INST_0_i_5
       (.I0(xfer_count_reg[9]),
        .I1(xfer_count_reg[8]),
        .I2(xfer_count_reg[11]),
        .I3(xfer_count_reg[10]),
        .O(out_tlast_INST_0_i_5_n_0));
  LUT4 #(
    .INIT(16'hFFA8)) 
    out_tvalid_INST_0
       (.I0(state__0[0]),
        .I1(had_in_tlast),
        .I2(in_tvalid),
        .I3(out_tvalid_INST_0_i_1_n_0),
        .O(out_tvalid));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT2 #(
    .INIT(4'hE)) 
    out_tvalid_INST_0_i_1
       (.I0(out_tvalid_INST_0_i_2_n_0),
        .I1(out_tvalid_INST_0_i_3_n_0),
        .O(out_tvalid_INST_0_i_1_n_0));
  LUT6 #(
    .INIT(64'hFFFFFFFFAAAAAAA8)) 
    out_tvalid_INST_0_i_2
       (.I0(state__0[0]),
        .I1(out_tvalid_INST_0_i_4_n_0),
        .I2(out_tvalid_INST_0_i_5_n_0),
        .I3(out_tvalid_INST_0_i_6_n_0),
        .I4(out_tvalid_INST_0_i_7_n_0),
        .I5(out_tvalid_INST_0_i_8_n_0),
        .O(out_tvalid_INST_0_i_2_n_0));
  LUT6 #(
    .INIT(64'h20200030303C003C)) 
    out_tvalid_INST_0_i_3
       (.I0(\out_tdata[63]_INST_0_i_7_n_0 ),
        .I1(state__0[2]),
        .I2(state__0[1]),
        .I3(state__0[0]),
        .I4(data_shift_next[2]),
        .I5(data_shift_next[1]),
        .O(out_tvalid_INST_0_i_3_n_0));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT5 #(
    .INIT(32'hBFFF0000)) 
    out_tvalid_INST_0_i_4
       (.I0(state__0[2]),
        .I1(\data_shift_reg_n_0_[7] ),
        .I2(data_shift_next[7]),
        .I3(data_shift_next[6]),
        .I4(state__0[1]),
        .O(out_tvalid_INST_0_i_4_n_0));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT4 #(
    .INIT(16'h0888)) 
    out_tvalid_INST_0_i_5
       (.I0(data_shift_next[5]),
        .I1(state__0[1]),
        .I2(data_shift_next[3]),
        .I3(data_shift_next[4]),
        .O(out_tvalid_INST_0_i_5_n_0));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT3 #(
    .INIT(8'h70)) 
    out_tvalid_INST_0_i_6
       (.I0(data_shift_next[4]),
        .I1(data_shift_next[3]),
        .I2(state__0[2]),
        .O(out_tvalid_INST_0_i_6_n_0));
  LUT6 #(
    .INIT(64'h4F4F4F4400000000)) 
    out_tvalid_INST_0_i_7
       (.I0(data_shift_next[2]),
        .I1(data_shift_next[1]),
        .I2(data_shift_next[5]),
        .I3(data_shift_next[4]),
        .I4(data_shift_next[3]),
        .I5(state__0[1]),
        .O(out_tvalid_INST_0_i_7_n_0));
  LUT6 #(
    .INIT(64'hEEE0EEE0EEF0EEE0)) 
    out_tvalid_INST_0_i_8
       (.I0(had_in_tlast),
        .I1(in_tvalid),
        .I2(state__0[2]),
        .I3(state__0[1]),
        .I4(data_shift_next[1]),
        .I5(data_shift_next[2]),
        .O(out_tvalid_INST_0_i_8_n_0));
  LUT6 #(
    .INIT(64'h888888888888F888)) 
    \xfer_count[0]_i_1 
       (.I0(out_tready),
        .I1(out_tvalid_INST_0_i_1_n_0),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[0]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'hEFFF)) 
    \xfer_count[0]_i_3 
       (.I0(state__0[1]),
        .I1(state__0[2]),
        .I2(state__0[0]),
        .I3(hdr_shift_next),
        .O(\xfer_count[0]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[0]_i_4 
       (.I0(in_tdata[3]),
        .I1(xfer_count_reg[3]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[0]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[0]_i_5 
       (.I0(in_tdata[2]),
        .I1(xfer_count_reg[2]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[0]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[0]_i_6 
       (.I0(in_tdata[1]),
        .I1(xfer_count_reg[1]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[0]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[0]_i_7 
       (.I0(in_tdata[0]),
        .I1(xfer_count_reg[0]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[0]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'h5755555554555555)) 
    \xfer_count[12]_i_2 
       (.I0(xfer_count_reg[15]),
        .I1(state__0[1]),
        .I2(state__0[2]),
        .I3(state__0[0]),
        .I4(hdr_shift_next),
        .I5(in_tdata[15]),
        .O(\xfer_count[12]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[12]_i_3 
       (.I0(in_tdata[14]),
        .I1(xfer_count_reg[14]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[12]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[12]_i_4 
       (.I0(in_tdata[13]),
        .I1(xfer_count_reg[13]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[12]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[12]_i_5 
       (.I0(in_tdata[12]),
        .I1(xfer_count_reg[12]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[12]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[4]_i_2 
       (.I0(in_tdata[7]),
        .I1(xfer_count_reg[7]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[4]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[4]_i_3 
       (.I0(in_tdata[6]),
        .I1(xfer_count_reg[6]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[4]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[4]_i_4 
       (.I0(in_tdata[5]),
        .I1(xfer_count_reg[5]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[4]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[4]_i_5 
       (.I0(in_tdata[4]),
        .I1(xfer_count_reg[4]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[4]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[8]_i_2 
       (.I0(in_tdata[11]),
        .I1(xfer_count_reg[11]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[8]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[8]_i_3 
       (.I0(in_tdata[10]),
        .I1(xfer_count_reg[10]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[8]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[8]_i_4 
       (.I0(in_tdata[9]),
        .I1(xfer_count_reg[9]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[8]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h333333333333A333)) 
    \xfer_count[8]_i_5 
       (.I0(in_tdata[8]),
        .I1(xfer_count_reg[8]),
        .I2(hdr_shift_next),
        .I3(state__0[0]),
        .I4(state__0[2]),
        .I5(state__0[1]),
        .O(\xfer_count[8]_i_5_n_0 ));
  FDRE \xfer_count_reg[0] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[0]_i_2_n_7 ),
        .Q(xfer_count_reg[0]),
        .R(rst));
  CARRY4 \xfer_count_reg[0]_i_2 
       (.CI(1'b0),
        .CO({\xfer_count_reg[0]_i_2_n_0 ,\xfer_count_reg[0]_i_2_n_1 ,\xfer_count_reg[0]_i_2_n_2 ,\xfer_count_reg[0]_i_2_n_3 }),
        .CYINIT(1'b0),
        .DI({\xfer_count[0]_i_3_n_0 ,\xfer_count[0]_i_3_n_0 ,\xfer_count[0]_i_3_n_0 ,\xfer_count[0]_i_3_n_0 }),
        .O({\xfer_count_reg[0]_i_2_n_4 ,\xfer_count_reg[0]_i_2_n_5 ,\xfer_count_reg[0]_i_2_n_6 ,\xfer_count_reg[0]_i_2_n_7 }),
        .S({\xfer_count[0]_i_4_n_0 ,\xfer_count[0]_i_5_n_0 ,\xfer_count[0]_i_6_n_0 ,\xfer_count[0]_i_7_n_0 }));
  FDRE \xfer_count_reg[10] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[8]_i_1_n_5 ),
        .Q(xfer_count_reg[10]),
        .R(rst));
  FDRE \xfer_count_reg[11] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[8]_i_1_n_4 ),
        .Q(xfer_count_reg[11]),
        .R(rst));
  FDRE \xfer_count_reg[12] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[12]_i_1_n_7 ),
        .Q(xfer_count_reg[12]),
        .R(rst));
  CARRY4 \xfer_count_reg[12]_i_1 
       (.CI(\xfer_count_reg[8]_i_1_n_0 ),
        .CO({\NLW_xfer_count_reg[12]_i_1_CO_UNCONNECTED [3],\xfer_count_reg[12]_i_1_n_1 ,\xfer_count_reg[12]_i_1_n_2 ,\xfer_count_reg[12]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,\xfer_count[0]_i_3_n_0 ,\xfer_count[0]_i_3_n_0 ,\xfer_count[0]_i_3_n_0 }),
        .O({\xfer_count_reg[12]_i_1_n_4 ,\xfer_count_reg[12]_i_1_n_5 ,\xfer_count_reg[12]_i_1_n_6 ,\xfer_count_reg[12]_i_1_n_7 }),
        .S({\xfer_count[12]_i_2_n_0 ,\xfer_count[12]_i_3_n_0 ,\xfer_count[12]_i_4_n_0 ,\xfer_count[12]_i_5_n_0 }));
  FDRE \xfer_count_reg[13] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[12]_i_1_n_6 ),
        .Q(xfer_count_reg[13]),
        .R(rst));
  FDRE \xfer_count_reg[14] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[12]_i_1_n_5 ),
        .Q(xfer_count_reg[14]),
        .R(rst));
  FDRE \xfer_count_reg[15] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[12]_i_1_n_4 ),
        .Q(xfer_count_reg[15]),
        .R(rst));
  FDRE \xfer_count_reg[1] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[0]_i_2_n_6 ),
        .Q(xfer_count_reg[1]),
        .R(rst));
  FDRE \xfer_count_reg[2] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[0]_i_2_n_5 ),
        .Q(xfer_count_reg[2]),
        .R(rst));
  FDRE \xfer_count_reg[3] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[0]_i_2_n_4 ),
        .Q(xfer_count_reg[3]),
        .R(rst));
  FDRE \xfer_count_reg[4] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[4]_i_1_n_7 ),
        .Q(xfer_count_reg[4]),
        .R(rst));
  CARRY4 \xfer_count_reg[4]_i_1 
       (.CI(\xfer_count_reg[0]_i_2_n_0 ),
        .CO({\xfer_count_reg[4]_i_1_n_0 ,\xfer_count_reg[4]_i_1_n_1 ,\xfer_count_reg[4]_i_1_n_2 ,\xfer_count_reg[4]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({\xfer_count[0]_i_3_n_0 ,\xfer_count[0]_i_3_n_0 ,\xfer_count[0]_i_3_n_0 ,\xfer_count[0]_i_3_n_0 }),
        .O({\xfer_count_reg[4]_i_1_n_4 ,\xfer_count_reg[4]_i_1_n_5 ,\xfer_count_reg[4]_i_1_n_6 ,\xfer_count_reg[4]_i_1_n_7 }),
        .S({\xfer_count[4]_i_2_n_0 ,\xfer_count[4]_i_3_n_0 ,\xfer_count[4]_i_4_n_0 ,\xfer_count[4]_i_5_n_0 }));
  FDRE \xfer_count_reg[5] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[4]_i_1_n_6 ),
        .Q(xfer_count_reg[5]),
        .R(rst));
  FDRE \xfer_count_reg[6] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[4]_i_1_n_5 ),
        .Q(xfer_count_reg[6]),
        .R(rst));
  FDRE \xfer_count_reg[7] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[4]_i_1_n_4 ),
        .Q(xfer_count_reg[7]),
        .R(rst));
  FDRE \xfer_count_reg[8] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[8]_i_1_n_7 ),
        .Q(xfer_count_reg[8]),
        .R(rst));
  CARRY4 \xfer_count_reg[8]_i_1 
       (.CI(\xfer_count_reg[4]_i_1_n_0 ),
        .CO({\xfer_count_reg[8]_i_1_n_0 ,\xfer_count_reg[8]_i_1_n_1 ,\xfer_count_reg[8]_i_1_n_2 ,\xfer_count_reg[8]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({\xfer_count[0]_i_3_n_0 ,\xfer_count[0]_i_3_n_0 ,\xfer_count[0]_i_3_n_0 ,\xfer_count[0]_i_3_n_0 }),
        .O({\xfer_count_reg[8]_i_1_n_4 ,\xfer_count_reg[8]_i_1_n_5 ,\xfer_count_reg[8]_i_1_n_6 ,\xfer_count_reg[8]_i_1_n_7 }),
        .S({\xfer_count[8]_i_2_n_0 ,\xfer_count[8]_i_3_n_0 ,\xfer_count[8]_i_4_n_0 ,\xfer_count[8]_i_5_n_0 }));
  FDRE \xfer_count_reg[9] 
       (.C(clk),
        .CE(\xfer_count[0]_i_1_n_0 ),
        .D(\xfer_count_reg[8]_i_1_n_6 ),
        .Q(xfer_count_reg[9]),
        .R(rst));
endmodule
`ifndef GLBL
`define GLBL
`timescale  1 ps / 1 ps

module glbl ();

    parameter ROC_WIDTH = 100000;
    parameter TOC_WIDTH = 0;

//--------   STARTUP Globals --------------
    wire GSR;
    wire GTS;
    wire GWE;
    wire PRLD;
    tri1 p_up_tmp;
    tri (weak1, strong0) PLL_LOCKG = p_up_tmp;

    wire PROGB_GLBL;
    wire CCLKO_GLBL;
    wire FCSBO_GLBL;
    wire [3:0] DO_GLBL;
    wire [3:0] DI_GLBL;
   
    reg GSR_int;
    reg GTS_int;
    reg PRLD_int;

//--------   JTAG Globals --------------
    wire JTAG_TDO_GLBL;
    wire JTAG_TCK_GLBL;
    wire JTAG_TDI_GLBL;
    wire JTAG_TMS_GLBL;
    wire JTAG_TRST_GLBL;

    reg JTAG_CAPTURE_GLBL;
    reg JTAG_RESET_GLBL;
    reg JTAG_SHIFT_GLBL;
    reg JTAG_UPDATE_GLBL;
    reg JTAG_RUNTEST_GLBL;

    reg JTAG_SEL1_GLBL = 0;
    reg JTAG_SEL2_GLBL = 0 ;
    reg JTAG_SEL3_GLBL = 0;
    reg JTAG_SEL4_GLBL = 0;

    reg JTAG_USER_TDO1_GLBL = 1'bz;
    reg JTAG_USER_TDO2_GLBL = 1'bz;
    reg JTAG_USER_TDO3_GLBL = 1'bz;
    reg JTAG_USER_TDO4_GLBL = 1'bz;

    assign (strong1, weak0) GSR = GSR_int;
    assign (strong1, weak0) GTS = GTS_int;
    assign (weak1, weak0) PRLD = PRLD_int;

    initial begin
	GSR_int = 1'b1;
	PRLD_int = 1'b1;
	#(ROC_WIDTH)
	GSR_int = 1'b0;
	PRLD_int = 1'b0;
    end

    initial begin
	GTS_int = 1'b1;
	#(TOC_WIDTH)
	GTS_int = 1'b0;
    end

endmodule
`endif
