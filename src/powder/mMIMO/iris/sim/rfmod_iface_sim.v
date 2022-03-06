// Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2018.3 (lin64) Build 2405991 Thu Dec  6 23:36:41 MST 2018
// Date        : Wed Nov 18 10:57:26 2020
// Host        : bender.ad.sklk.us running 64-bit Ubuntu 16.04.6 LTS
// Command     : write_verilog -force -mode funcsim rfmod_iface_sim.v
// Design      : rfmod_iface
// Purpose     : This verilog netlist is a functional simulation representation of the design and should not be modified
//               or synthesized. This netlist cannot be used for SDF annotated simulation.
// Device      : xc7z030sbg485-1
// --------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

(* CLK_FREQ = "100000000" *) (* ID_FE_01_DEV = "4'b1111" *) (* ID_FE_02_UHF = "4'b0001" *) 
(* ID_FE_03_CBRS = "4'b0010" *) (* dut_cbrs_mask = "16'b1111100001100000" *) (* dut_dev_mask = "16'b0001100111110000" *) 
(* dut_uhf_mask = "16'b1111100011100000" *) (* test_dn_mask = "16'b0000111100001111" *) (* test_up_mask = "16'b1111000011110000" *) 
(* NotValidForBitStream *)
module rfmod_iface
   (clk,
    rst,
    sync,
    dut_pgood,
    tx_active,
    rfmod_in,
    rfmod_out,
    rfmod_oe,
    rfmod_id,
    id_valid,
    gain_override_out,
    agc_en,
    \gain_lna1_out[0] ,
    \gain_lna1_out[1] ,
    \gain_lna2_out[0] ,
    \gain_lna2_out[1] ,
    \gain_attn_out[0] ,
    \gain_attn_out[1] ,
    dut_tdi,
    dut_tdo,
    dut_tck,
    dut_tms,
    dut_ten,
    addr,
    dati,
    dato,
    wr,
    en,
    rdy,
    led_uplink_on,
    led_dnlink_on,
    led_good,
    led_error,
    debug,
    cbrs_rev);
  input clk;
  input rst;
  input sync;
  input dut_pgood;
  input [1:0]tx_active;
  input [16:1]rfmod_in;
  output [16:1]rfmod_out;
  output [16:1]rfmod_oe;
  output [3:0]rfmod_id;
  output id_valid;
  input gain_override_out;
  input agc_en;
  input [3:0]\gain_lna1_out[0] ;
  input [3:0]\gain_lna1_out[1] ;
  input [15:0]\gain_lna2_out[0] ;
  input [15:0]\gain_lna2_out[1] ;
  input [15:0]\gain_attn_out[0] ;
  input [15:0]\gain_attn_out[1] ;
  input dut_tdi;
  output dut_tdo;
  input dut_tck;
  input dut_tms;
  input dut_ten;
  input [7:0]addr;
  input [15:0]dati;
  output [15:0]dato;
  input wr;
  input en;
  output rdy;
  input led_uplink_on;
  input led_dnlink_on;
  input led_good;
  input led_error;
  output [7:0]debug;
  output [3:0]cbrs_rev;

  wire \<const0> ;
  wire \FSM_onehot_state[0]_i_1_n_0 ;
  wire \FSM_onehot_state[5]_i_1_n_0 ;
  wire \FSM_onehot_state[5]_i_2_n_0 ;
  wire \FSM_onehot_state[5]_i_3_n_0 ;
  wire \FSM_onehot_state[5]_i_4_n_0 ;
  wire \FSM_onehot_state[5]_i_5_n_0 ;
  wire \FSM_onehot_state_reg[0]_i_1_n_0 ;
  wire \FSM_onehot_state_reg[0]_i_2_n_0 ;
  wire \FSM_onehot_state_reg[0]_i_3_n_0 ;
  wire \FSM_onehot_state_reg[0]_i_4_n_0 ;
  wire \FSM_onehot_state_reg[10]_i_1_n_0 ;
  wire \FSM_onehot_state_reg[11]_i_2_n_0 ;
  wire \FSM_onehot_state_reg[11]_i_3_n_0 ;
  wire \FSM_onehot_state_reg[1]_i_1_n_0 ;
  wire \FSM_onehot_state_reg[1]_i_2_n_0 ;
  wire \FSM_onehot_state_reg[2]_i_1_n_0 ;
  wire \FSM_onehot_state_reg[2]_i_2_n_0 ;
  wire \FSM_onehot_state_reg[3]_i_1_n_0 ;
  wire \FSM_onehot_state_reg[4]_i_1_n_0 ;
  wire \FSM_onehot_state_reg[4]_i_2_n_0 ;
  wire \FSM_onehot_state_reg[5]_i_1_n_0 ;
  wire \FSM_onehot_state_reg[6]_i_1_n_0 ;
  wire \FSM_onehot_state_reg[6]_i_2_n_0 ;
  wire \FSM_onehot_state_reg[6]_i_3_n_0 ;
  wire \FSM_onehot_state_reg[6]_i_4_n_0 ;
  wire \FSM_onehot_state_reg[6]_i_5_n_0 ;
  wire \FSM_onehot_state_reg[6]_i_6_n_0 ;
  wire \FSM_onehot_state_reg[6]_i_7_n_0 ;
  wire \FSM_onehot_state_reg[6]_i_8_n_0 ;
  wire \FSM_onehot_state_reg[7]_i_1_n_0 ;
  wire \FSM_onehot_state_reg[7]_i_2_n_0 ;
  wire \FSM_onehot_state_reg[7]_i_3_n_0 ;
  wire \FSM_onehot_state_reg[7]_i_4_n_0 ;
  wire \FSM_onehot_state_reg[8]_i_1_n_0 ;
  wire \FSM_onehot_state_reg[9]_i_1_n_0 ;
  wire \FSM_onehot_state_reg[9]_i_2_n_0 ;
  wire \FSM_sequential_phy_state_reg[0]_i_1_n_0 ;
  wire \FSM_sequential_phy_state_reg[0]_i_2_n_0 ;
  wire \FSM_sequential_phy_state_reg[0]_i_3_n_0 ;
  wire \FSM_sequential_phy_state_reg[0]_i_4_n_0 ;
  wire \FSM_sequential_phy_state_reg[0]_i_5_n_0 ;
  wire \FSM_sequential_phy_state_reg[1]_i_1_n_0 ;
  wire \FSM_sequential_phy_state_reg[1]_i_2_n_0 ;
  wire \FSM_sequential_phy_state_reg[2]_i_1_n_0 ;
  wire \FSM_sequential_phy_state_reg[2]_i_2_n_0 ;
  wire \FSM_sequential_phy_state_reg[3]_i_10_n_0 ;
  wire \FSM_sequential_phy_state_reg[3]_i_2_n_0 ;
  wire \FSM_sequential_phy_state_reg[3]_i_3_n_0 ;
  wire \FSM_sequential_phy_state_reg[3]_i_4_n_0 ;
  wire \FSM_sequential_phy_state_reg[3]_i_5_n_0 ;
  wire \FSM_sequential_phy_state_reg[3]_i_6_n_0 ;
  wire \FSM_sequential_phy_state_reg[3]_i_7_n_0 ;
  wire \FSM_sequential_phy_state_reg[3]_i_8_n_0 ;
  wire \FSM_sequential_phy_state_reg[3]_i_9_n_0 ;
  wire \FSM_sequential_state[0]_i_1_n_0 ;
  wire \FSM_sequential_state[0]_i_2_n_0 ;
  wire \FSM_sequential_state[1]_i_1_n_0 ;
  wire \FSM_sequential_state[2]_i_1_n_0 ;
  wire \FSM_sequential_state[2]_i_2_n_0 ;
  wire \FSM_sequential_state[2]_i_3_n_0 ;
  wire \FSM_sequential_state[3]_i_1_n_0 ;
  wire \FSM_sequential_state[3]_i_2_n_0 ;
  wire \FSM_sequential_state[3]_i_3_n_0 ;
  wire \FSM_sequential_state[3]_i_4_n_0 ;
  wire \FSM_sequential_state[3]_i_5_n_0 ;
  wire \FSM_sequential_state[3]_i_6_n_0 ;
  wire \FSM_sequential_state[3]_i_7_n_0 ;
  wire __do_out1103_out;
  wire __do_out1108_out;
  wire __do_out1113_out;
  wire __do_out1118_out;
  wire __do_out1123_out;
  wire __do_out1127_out;
  wire __do_out183_out;
  wire __do_out188_out;
  wire __do_out193_out;
  wire \__do_out[0]_i_10_n_0 ;
  wire \__do_out[0]_i_11_n_0 ;
  wire \__do_out[0]_i_12_n_0 ;
  wire \__do_out[0]_i_13_n_0 ;
  wire \__do_out[0]_i_14_n_0 ;
  wire \__do_out[0]_i_15_n_0 ;
  wire \__do_out[0]_i_16_n_0 ;
  wire \__do_out[0]_i_17_n_0 ;
  wire \__do_out[0]_i_18_n_0 ;
  wire \__do_out[0]_i_2_n_0 ;
  wire \__do_out[0]_i_3_n_0 ;
  wire \__do_out[0]_i_4_n_0 ;
  wire \__do_out[0]_i_5_n_0 ;
  wire \__do_out[0]_i_6_n_0 ;
  wire \__do_out[0]_i_7_n_0 ;
  wire \__do_out[0]_i_8_n_0 ;
  wire \__do_out[0]_i_9_n_0 ;
  wire \__do_out[10]_i_10_n_0 ;
  wire \__do_out[10]_i_11_n_0 ;
  wire \__do_out[10]_i_2_n_0 ;
  wire \__do_out[10]_i_3_n_0 ;
  wire \__do_out[10]_i_4_n_0 ;
  wire \__do_out[10]_i_5_n_0 ;
  wire \__do_out[10]_i_6_n_0 ;
  wire \__do_out[10]_i_7_n_0 ;
  wire \__do_out[10]_i_8_n_0 ;
  wire \__do_out[10]_i_9_n_0 ;
  wire \__do_out[11]_i_10_n_0 ;
  wire \__do_out[11]_i_2_n_0 ;
  wire \__do_out[11]_i_3_n_0 ;
  wire \__do_out[11]_i_4_n_0 ;
  wire \__do_out[11]_i_5_n_0 ;
  wire \__do_out[11]_i_6_n_0 ;
  wire \__do_out[11]_i_7_n_0 ;
  wire \__do_out[11]_i_8_n_0 ;
  wire \__do_out[11]_i_9_n_0 ;
  wire \__do_out[12]_i_10_n_0 ;
  wire \__do_out[12]_i_2_n_0 ;
  wire \__do_out[12]_i_3_n_0 ;
  wire \__do_out[12]_i_4_n_0 ;
  wire \__do_out[12]_i_5_n_0 ;
  wire \__do_out[12]_i_6_n_0 ;
  wire \__do_out[12]_i_7_n_0 ;
  wire \__do_out[12]_i_8_n_0 ;
  wire \__do_out[12]_i_9_n_0 ;
  wire \__do_out[13]_i_10_n_0 ;
  wire \__do_out[13]_i_11_n_0 ;
  wire \__do_out[13]_i_12_n_0 ;
  wire \__do_out[13]_i_13_n_0 ;
  wire \__do_out[13]_i_14_n_0 ;
  wire \__do_out[13]_i_15_n_0 ;
  wire \__do_out[13]_i_16_n_0 ;
  wire \__do_out[13]_i_17_n_0 ;
  wire \__do_out[13]_i_2_n_0 ;
  wire \__do_out[13]_i_3_n_0 ;
  wire \__do_out[13]_i_4_n_0 ;
  wire \__do_out[13]_i_5_n_0 ;
  wire \__do_out[13]_i_6_n_0 ;
  wire \__do_out[13]_i_7_n_0 ;
  wire \__do_out[13]_i_8_n_0 ;
  wire \__do_out[13]_i_9_n_0 ;
  wire \__do_out[14]_i_10_n_0 ;
  wire \__do_out[14]_i_11_n_0 ;
  wire \__do_out[14]_i_12_n_0 ;
  wire \__do_out[14]_i_13_n_0 ;
  wire \__do_out[14]_i_14_n_0 ;
  wire \__do_out[14]_i_15_n_0 ;
  wire \__do_out[14]_i_16_n_0 ;
  wire \__do_out[14]_i_17_n_0 ;
  wire \__do_out[14]_i_18_n_0 ;
  wire \__do_out[14]_i_19_n_0 ;
  wire \__do_out[14]_i_20_n_0 ;
  wire \__do_out[14]_i_21_n_0 ;
  wire \__do_out[14]_i_22_n_0 ;
  wire \__do_out[14]_i_23_n_0 ;
  wire \__do_out[14]_i_24_n_0 ;
  wire \__do_out[14]_i_2_n_0 ;
  wire \__do_out[14]_i_3_n_0 ;
  wire \__do_out[14]_i_4_n_0 ;
  wire \__do_out[14]_i_5_n_0 ;
  wire \__do_out[14]_i_6_n_0 ;
  wire \__do_out[14]_i_7_n_0 ;
  wire \__do_out[14]_i_8_n_0 ;
  wire \__do_out[14]_i_9_n_0 ;
  wire \__do_out[15]_i_10_n_0 ;
  wire \__do_out[15]_i_11_n_0 ;
  wire \__do_out[15]_i_12_n_0 ;
  wire \__do_out[15]_i_13_n_0 ;
  wire \__do_out[15]_i_14_n_0 ;
  wire \__do_out[15]_i_15_n_0 ;
  wire \__do_out[15]_i_16_n_0 ;
  wire \__do_out[15]_i_17_n_0 ;
  wire \__do_out[15]_i_18_n_0 ;
  wire \__do_out[15]_i_19_n_0 ;
  wire \__do_out[15]_i_1_n_0 ;
  wire \__do_out[15]_i_20_n_0 ;
  wire \__do_out[15]_i_21_n_0 ;
  wire \__do_out[15]_i_22_n_0 ;
  wire \__do_out[15]_i_23_n_0 ;
  wire \__do_out[15]_i_24_n_0 ;
  wire \__do_out[15]_i_25_n_0 ;
  wire \__do_out[15]_i_26_n_0 ;
  wire \__do_out[15]_i_27_n_0 ;
  wire \__do_out[15]_i_28_n_0 ;
  wire \__do_out[15]_i_29_n_0 ;
  wire \__do_out[15]_i_30_n_0 ;
  wire \__do_out[15]_i_31_n_0 ;
  wire \__do_out[15]_i_32_n_0 ;
  wire \__do_out[15]_i_33_n_0 ;
  wire \__do_out[15]_i_34_n_0 ;
  wire \__do_out[15]_i_35_n_0 ;
  wire \__do_out[15]_i_3_n_0 ;
  wire \__do_out[15]_i_4_n_0 ;
  wire \__do_out[15]_i_5_n_0 ;
  wire \__do_out[15]_i_6_n_0 ;
  wire \__do_out[15]_i_7_n_0 ;
  wire \__do_out[15]_i_8_n_0 ;
  wire \__do_out[15]_i_9_n_0 ;
  wire \__do_out[1]_i_10_n_0 ;
  wire \__do_out[1]_i_11_n_0 ;
  wire \__do_out[1]_i_12_n_0 ;
  wire \__do_out[1]_i_13_n_0 ;
  wire \__do_out[1]_i_14_n_0 ;
  wire \__do_out[1]_i_15_n_0 ;
  wire \__do_out[1]_i_16_n_0 ;
  wire \__do_out[1]_i_17_n_0 ;
  wire \__do_out[1]_i_2_n_0 ;
  wire \__do_out[1]_i_3_n_0 ;
  wire \__do_out[1]_i_4_n_0 ;
  wire \__do_out[1]_i_5_n_0 ;
  wire \__do_out[1]_i_6_n_0 ;
  wire \__do_out[1]_i_7_n_0 ;
  wire \__do_out[1]_i_8_n_0 ;
  wire \__do_out[1]_i_9_n_0 ;
  wire \__do_out[2]_i_10_n_0 ;
  wire \__do_out[2]_i_11_n_0 ;
  wire \__do_out[2]_i_12_n_0 ;
  wire \__do_out[2]_i_13_n_0 ;
  wire \__do_out[2]_i_14_n_0 ;
  wire \__do_out[2]_i_15_n_0 ;
  wire \__do_out[2]_i_16_n_0 ;
  wire \__do_out[2]_i_17_n_0 ;
  wire \__do_out[2]_i_2_n_0 ;
  wire \__do_out[2]_i_3_n_0 ;
  wire \__do_out[2]_i_4_n_0 ;
  wire \__do_out[2]_i_5_n_0 ;
  wire \__do_out[2]_i_6_n_0 ;
  wire \__do_out[2]_i_7_n_0 ;
  wire \__do_out[2]_i_8_n_0 ;
  wire \__do_out[2]_i_9_n_0 ;
  wire \__do_out[3]_i_10_n_0 ;
  wire \__do_out[3]_i_11_n_0 ;
  wire \__do_out[3]_i_12_n_0 ;
  wire \__do_out[3]_i_13_n_0 ;
  wire \__do_out[3]_i_14_n_0 ;
  wire \__do_out[3]_i_15_n_0 ;
  wire \__do_out[3]_i_16_n_0 ;
  wire \__do_out[3]_i_17_n_0 ;
  wire \__do_out[3]_i_2_n_0 ;
  wire \__do_out[3]_i_3_n_0 ;
  wire \__do_out[3]_i_4_n_0 ;
  wire \__do_out[3]_i_5_n_0 ;
  wire \__do_out[3]_i_6_n_0 ;
  wire \__do_out[3]_i_7_n_0 ;
  wire \__do_out[3]_i_8_n_0 ;
  wire \__do_out[3]_i_9_n_0 ;
  wire \__do_out[4]_i_10_n_0 ;
  wire \__do_out[4]_i_11_n_0 ;
  wire \__do_out[4]_i_12_n_0 ;
  wire \__do_out[4]_i_13_n_0 ;
  wire \__do_out[4]_i_14_n_0 ;
  wire \__do_out[4]_i_15_n_0 ;
  wire \__do_out[4]_i_16_n_0 ;
  wire \__do_out[4]_i_17_n_0 ;
  wire \__do_out[4]_i_2_n_0 ;
  wire \__do_out[4]_i_3_n_0 ;
  wire \__do_out[4]_i_4_n_0 ;
  wire \__do_out[4]_i_5_n_0 ;
  wire \__do_out[4]_i_6_n_0 ;
  wire \__do_out[4]_i_7_n_0 ;
  wire \__do_out[4]_i_8_n_0 ;
  wire \__do_out[4]_i_9_n_0 ;
  wire \__do_out[5]_i_10_n_0 ;
  wire \__do_out[5]_i_11_n_0 ;
  wire \__do_out[5]_i_2_n_0 ;
  wire \__do_out[5]_i_3_n_0 ;
  wire \__do_out[5]_i_4_n_0 ;
  wire \__do_out[5]_i_5_n_0 ;
  wire \__do_out[5]_i_6_n_0 ;
  wire \__do_out[5]_i_7_n_0 ;
  wire \__do_out[5]_i_8_n_0 ;
  wire \__do_out[5]_i_9_n_0 ;
  wire \__do_out[6]_i_10_n_0 ;
  wire \__do_out[6]_i_11_n_0 ;
  wire \__do_out[6]_i_12_n_0 ;
  wire \__do_out[6]_i_13_n_0 ;
  wire \__do_out[6]_i_14_n_0 ;
  wire \__do_out[6]_i_2_n_0 ;
  wire \__do_out[6]_i_3_n_0 ;
  wire \__do_out[6]_i_4_n_0 ;
  wire \__do_out[6]_i_5_n_0 ;
  wire \__do_out[6]_i_6_n_0 ;
  wire \__do_out[6]_i_7_n_0 ;
  wire \__do_out[6]_i_8_n_0 ;
  wire \__do_out[6]_i_9_n_0 ;
  wire \__do_out[7]_i_10_n_0 ;
  wire \__do_out[7]_i_11_n_0 ;
  wire \__do_out[7]_i_12_n_0 ;
  wire \__do_out[7]_i_13_n_0 ;
  wire \__do_out[7]_i_14_n_0 ;
  wire \__do_out[7]_i_15_n_0 ;
  wire \__do_out[7]_i_16_n_0 ;
  wire \__do_out[7]_i_2_n_0 ;
  wire \__do_out[7]_i_3_n_0 ;
  wire \__do_out[7]_i_4_n_0 ;
  wire \__do_out[7]_i_5_n_0 ;
  wire \__do_out[7]_i_6_n_0 ;
  wire \__do_out[7]_i_7_n_0 ;
  wire \__do_out[7]_i_8_n_0 ;
  wire \__do_out[7]_i_9_n_0 ;
  wire \__do_out[8]_i_10_n_0 ;
  wire \__do_out[8]_i_2_n_0 ;
  wire \__do_out[8]_i_3_n_0 ;
  wire \__do_out[8]_i_4_n_0 ;
  wire \__do_out[8]_i_5_n_0 ;
  wire \__do_out[8]_i_6_n_0 ;
  wire \__do_out[8]_i_7_n_0 ;
  wire \__do_out[8]_i_8_n_0 ;
  wire \__do_out[8]_i_9_n_0 ;
  wire \__do_out[9]_i_10_n_0 ;
  wire \__do_out[9]_i_11_n_0 ;
  wire \__do_out[9]_i_2_n_0 ;
  wire \__do_out[9]_i_3_n_0 ;
  wire \__do_out[9]_i_4_n_0 ;
  wire \__do_out[9]_i_5_n_0 ;
  wire \__do_out[9]_i_6_n_0 ;
  wire \__do_out[9]_i_7_n_0 ;
  wire \__do_out[9]_i_8_n_0 ;
  wire \__do_out[9]_i_9_n_0 ;
  wire [7:0]addr;
  wire \addr_reg[6]_i_1_n_0 ;
  wire \addr_reg[6]_i_2_n_0 ;
  wire agc_en;
  wire \bit_count_reg[3]_i_1_n_0 ;
  wire \bit_count_reg[3]_i_3_n_0 ;
  wire \bit_count_reg[3]_i_4_n_0 ;
  wire \bit_count_reg[3]_i_5_n_0 ;
  wire \bit_count_reg[3]_i_6_n_0 ;
  wire bus_active_reg_i_1_n_0;
  wire [0:0]\^cbrs_rev ;
  wire clk;
  wire cmd_ready_reg_i_1_n_0;
  wire \count[6]_i_2_n_0 ;
  wire [15:0]ctrl0_rd;
  wire \ctrl0_reg_n_0_[0] ;
  wire \ctrl0_reg_n_0_[10] ;
  wire \ctrl0_reg_n_0_[11] ;
  wire \ctrl0_reg_n_0_[12] ;
  wire \ctrl0_reg_n_0_[13] ;
  wire \ctrl0_reg_n_0_[14] ;
  wire \ctrl0_reg_n_0_[4] ;
  wire \ctrl0_reg_n_0_[5] ;
  wire \ctrl0_reg_n_0_[6] ;
  wire \ctrl0_reg_n_0_[7] ;
  wire \ctrl0_reg_n_0_[8] ;
  wire \ctrl0_reg_n_0_[9] ;
  wire [15:0]ctrl1_rd;
  wire \ctrl1_rd[0]_i_1_n_0 ;
  wire \ctrl1_rd[10]_i_1_n_0 ;
  wire \ctrl1_rd[11]_i_1_n_0 ;
  wire \ctrl1_rd[12]_i_1_n_0 ;
  wire \ctrl1_rd[13]_i_1_n_0 ;
  wire \ctrl1_rd[14]_i_1_n_0 ;
  wire \ctrl1_rd[15]_i_1_n_0 ;
  wire \ctrl1_rd[1]_i_1_n_0 ;
  wire \ctrl1_rd[2]_i_1_n_0 ;
  wire \ctrl1_rd[3]_i_1_n_0 ;
  wire \ctrl1_rd[4]_i_1_n_0 ;
  wire \ctrl1_rd[5]_i_1_n_0 ;
  wire \ctrl1_rd[6]_i_1_n_0 ;
  wire \ctrl1_rd[7]_i_1_n_0 ;
  wire \ctrl1_rd[8]_i_1_n_0 ;
  wire \ctrl1_rd[9]_i_1_n_0 ;
  wire \ctrl1_reg_n_0_[0] ;
  wire \ctrl1_reg_n_0_[10] ;
  wire \ctrl1_reg_n_0_[11] ;
  wire \ctrl1_reg_n_0_[12] ;
  wire \ctrl1_reg_n_0_[14] ;
  wire \ctrl1_reg_n_0_[15] ;
  wire \ctrl1_reg_n_0_[1] ;
  wire \ctrl1_reg_n_0_[2] ;
  wire \ctrl1_reg_n_0_[3] ;
  wire \ctrl1_reg_n_0_[4] ;
  wire \ctrl1_reg_n_0_[5] ;
  wire \ctrl1_reg_n_0_[6] ;
  wire \ctrl1_reg_n_0_[7] ;
  wire \ctrl1_reg_n_0_[8] ;
  wire \ctrl1_reg_n_0_[9] ;
  wire [15:0]ctrl2_rd;
  wire \ctrl2_rd[0]_i_1_n_0 ;
  wire \ctrl2_rd[1]_i_1_n_0 ;
  wire \ctrl2_rd[2]_i_1_n_0 ;
  wire \ctrl2_rd[3]_i_1_n_0 ;
  wire \ctrl2_reg_n_0_[0] ;
  wire \ctrl2_reg_n_0_[10] ;
  wire \ctrl2_reg_n_0_[11] ;
  wire \ctrl2_reg_n_0_[12] ;
  wire \ctrl2_reg_n_0_[13] ;
  wire \ctrl2_reg_n_0_[14] ;
  wire \ctrl2_reg_n_0_[15] ;
  wire \ctrl2_reg_n_0_[1] ;
  wire \ctrl2_reg_n_0_[2] ;
  wire \ctrl2_reg_n_0_[3] ;
  wire \ctrl2_reg_n_0_[4] ;
  wire \ctrl2_reg_n_0_[5] ;
  wire \ctrl2_reg_n_0_[6] ;
  wire \ctrl2_reg_n_0_[7] ;
  wire \ctrl2_reg_n_0_[8] ;
  wire \ctrl2_reg_n_0_[9] ;
  wire \ctrl3_reg_n_0_[0] ;
  wire \ctrl3_reg_n_0_[10] ;
  wire \ctrl3_reg_n_0_[11] ;
  wire \ctrl3_reg_n_0_[12] ;
  wire \ctrl3_reg_n_0_[13] ;
  wire \ctrl3_reg_n_0_[14] ;
  wire \ctrl3_reg_n_0_[15] ;
  wire \ctrl3_reg_n_0_[1] ;
  wire \ctrl3_reg_n_0_[2] ;
  wire \ctrl3_reg_n_0_[3] ;
  wire \ctrl3_reg_n_0_[4] ;
  wire \ctrl3_reg_n_0_[5] ;
  wire \ctrl3_reg_n_0_[6] ;
  wire \ctrl3_reg_n_0_[7] ;
  wire \ctrl3_reg_n_0_[8] ;
  wire \ctrl3_reg_n_0_[9] ;
  wire dat_cache1_carry__0_i_1_n_0;
  wire dat_cache1_carry__0_i_2_n_0;
  wire dat_cache1_carry__0_i_3_n_0;
  wire dat_cache1_carry__0_i_4_n_0;
  wire dat_cache1_carry__1_i_1_n_0;
  wire dat_cache1_carry__1_i_2_n_0;
  wire dat_cache1_carry__1_i_3_n_0;
  wire dat_cache1_carry_i_1_n_0;
  wire dat_cache1_carry_i_2_n_0;
  wire dat_cache1_carry_i_3_n_0;
  wire dat_cache1_carry_i_4_n_0;
  wire dat_cache1_carry_i_5_n_0;
  wire \dat_cache[21]_i_2_n_0 ;
  wire \dat_cache[25]_i_2_n_0 ;
  wire data0;
  wire data_in_ready_reg_i_1_n_0;
  wire \data_o0[7]_i_1__0_n_0 ;
  wire \data_o0[7]_i_1_n_0 ;
  wire \data_o1[7]_i_1_n_0 ;
  wire data_out_last_reg_i_1_n_0;
  wire \data_out_reg[7]_i_1_n_0 ;
  wire data_out_valid_reg_i_1_n_0;
  wire data_out_valid_reg_i_2_n_0;
  wire data_out_valid_reg_i_3_n_0;
  wire \data_reg[0]_i_1_n_0 ;
  wire \data_reg[1]_i_1_n_0 ;
  wire \data_reg[2]_i_1_n_0 ;
  wire \data_reg[3]_i_1_n_0 ;
  wire \data_reg[4]_i_1_n_0 ;
  wire \data_reg[5]_i_1_n_0 ;
  wire \data_reg[6]_i_1_n_0 ;
  wire \data_reg[7]_i_1_n_0 ;
  wire \data_reg[7]_i_2_n_0 ;
  wire [15:0]dati;
  wire [15:0]dato;
  wire delay_next0_carry__0_i_1_n_0;
  wire delay_next0_carry__0_i_2_n_0;
  wire delay_next0_carry__0_i_3_n_0;
  wire delay_next0_carry__0_i_4_n_0;
  wire delay_next0_carry__1_i_1_n_0;
  wire delay_next0_carry__1_i_2_n_0;
  wire delay_next0_carry__1_i_3_n_0;
  wire delay_next0_carry__1_i_4_n_0;
  wire delay_next0_carry__2_i_1_n_0;
  wire delay_next0_carry__2_i_2_n_0;
  wire delay_next0_carry__2_i_3_n_0;
  wire delay_next0_carry__2_i_4_n_0;
  wire delay_next0_carry_i_1_n_0;
  wire delay_next0_carry_i_2_n_0;
  wire delay_next0_carry_i_3_n_0;
  wire delay_next0_carry_i_4_n_0;
  wire \delay_reg[0]_i_1_n_0 ;
  wire \delay_reg[10]_i_1_n_0 ;
  wire \delay_reg[11]_i_1_n_0 ;
  wire \delay_reg[12]_i_1_n_0 ;
  wire \delay_reg[13]_i_1_n_0 ;
  wire \delay_reg[14]_i_1_n_0 ;
  wire \delay_reg[15]_i_1_n_0 ;
  wire \delay_reg[16]_i_1_n_0 ;
  wire \delay_reg[16]_i_2_n_0 ;
  wire \delay_reg[16]_i_3_n_0 ;
  wire \delay_reg[16]_i_4_n_0 ;
  wire \delay_reg[1]_i_1_n_0 ;
  wire \delay_reg[2]_i_1_n_0 ;
  wire \delay_reg[3]_i_1_n_0 ;
  wire \delay_reg[4]_i_1_n_0 ;
  wire \delay_reg[5]_i_1_n_0 ;
  wire \delay_reg[6]_i_1_n_0 ;
  wire \delay_reg[7]_i_1_n_0 ;
  wire \delay_reg[8]_i_1_n_0 ;
  wire \delay_reg[9]_i_1_n_0 ;
  wire delay_scl_reg_i_1_n_0;
  wire delay_scl_reg_i_2_n_0;
  wire dut_pgood;
  wire dut_tck;
  wire dut_tdi;
  wire dut_tdo;
  wire dut_ten;
  wire dut_tms;
  wire en;
  wire en_i_1_n_0;
  wire \err[0]_i_1_n_0 ;
  wire \err[1]_i_1_n_0 ;
  wire \err[2]_i_1_n_0 ;
  wire \err[3]_i_1_n_0 ;
  wire \err[4]_i_1_n_0 ;
  wire \err[5]_i_1_n_0 ;
  wire \err[6]_i_1_n_0 ;
  wire \err[7]_i_1_n_0 ;
  wire \err[7]_i_3_n_0 ;
  wire error__0_i_1_n_0;
  wire error_i_1_n_0;
  wire ext_cmd_queued_i_1_n_0;
  wire fdd_en_b;
  wire [1:0]gain_attn2_local;
  wire \gain_attn2_local[0]_i_1_n_0 ;
  wire \gain_attn2_local[1]_i_1_n_0 ;
  wire [15:0]\gain_attn_out[0] ;
  wire [15:0]\gain_attn_out[1] ;
  wire [3:0]\gain_lna1_out[0] ;
  wire [15:0]\gain_lna2_out[0] ;
  wire \gpio_dato_reg_n_0_[10] ;
  wire \gpio_dato_reg_n_0_[11] ;
  wire \gpio_dato_reg_n_0_[12] ;
  wire \gpio_dato_reg_n_0_[13] ;
  wire \gpio_dato_reg_n_0_[14] ;
  wire \gpio_dato_reg_n_0_[15] ;
  wire \gpio_dato_reg_n_0_[16] ;
  wire \gpio_dato_reg_n_0_[1] ;
  wire \gpio_dato_reg_n_0_[2] ;
  wire \gpio_dato_reg_n_0_[3] ;
  wire \gpio_dato_reg_n_0_[4] ;
  wire \gpio_dato_reg_n_0_[5] ;
  wire \gpio_dato_reg_n_0_[7] ;
  wire \gpio_dato_reg_n_0_[8] ;
  wire \gpio_dato_reg_n_0_[9] ;
  wire \gpio_dir_reg_n_0_[10] ;
  wire \gpio_dir_reg_n_0_[11] ;
  wire \gpio_dir_reg_n_0_[12] ;
  wire \gpio_dir_reg_n_0_[13] ;
  wire \gpio_dir_reg_n_0_[14] ;
  wire \gpio_dir_reg_n_0_[15] ;
  wire \gpio_dir_reg_n_0_[16] ;
  wire \gpio_dir_reg_n_0_[1] ;
  wire \gpio_dir_reg_n_0_[3] ;
  wire \gpio_dir_reg_n_0_[4] ;
  wire \gpio_dir_reg_n_0_[6] ;
  wire \gpio_dir_reg_n_0_[7] ;
  wire \gpio_dir_reg_n_0_[8] ;
  wire \gpio_dir_reg_n_0_[9] ;
  wire gpio_exp_busy;
  wire [31:0]gpio_exp_out;
  wire [31:1]gpio_exp_rb;
  wire gpio_exp_trigger;
  wire gpio_exp_trigger0;
  wire gpio_exp_trigger_i_2_n_0;
  wire [6:0]i2c_addr_s;
  wire i2c_addr_s0;
  wire \i2c_addr_s[0]_i_1_n_0 ;
  wire \i2c_addr_s[1]_i_1_n_0 ;
  wire \i2c_addr_s[2]_i_1_n_0 ;
  wire \i2c_addr_s[3]_i_1_n_0 ;
  wire \i2c_addr_s[4]_i_1_n_0 ;
  wire \i2c_addr_s[5]_i_1_n_0 ;
  wire \i2c_addr_s[6]_i_1_n_0 ;
  wire \i2c_addr_s[6]_i_2_n_0 ;
  wire \i2c_addr_s[6]_i_3_n_0 ;
  wire \i2c_addr_s[6]_i_4_n_0 ;
  wire \i2c_addr_s[6]_i_5_n_0 ;
  wire \i2c_addr_s[6]_i_6_n_0 ;
  wire \i2c_addr_s[6]_i_7_n_0 ;
  wire i2c_busy;
  wire [2:0]i2c_cache_addr;
  wire \i2c_cache_addr[0]_i_1_n_0 ;
  wire \i2c_cache_addr[1]_i_1_n_0 ;
  wire \i2c_cache_addr[2]_i_1_n_0 ;
  wire i2c_cmd_en;
  wire i2c_cmd_en0;
  wire [3:0]i2c_cmd_mode;
  wire [7:0]i2c_data_i0;
  wire \i2c_data_i0[0]_i_1_n_0 ;
  wire \i2c_data_i0[1]_i_1_n_0 ;
  wire \i2c_data_i0[2]_i_1_n_0 ;
  wire \i2c_data_i0[3]_i_1_n_0 ;
  wire \i2c_data_i0[4]_i_1_n_0 ;
  wire \i2c_data_i0[5]_i_1_n_0 ;
  wire \i2c_data_i0[6]_i_1_n_0 ;
  wire \i2c_data_i0[7]_i_1_n_0 ;
  wire [7:0]i2c_data_i1;
  wire \i2c_data_i1[0]_i_1_n_0 ;
  wire \i2c_data_i1[1]_i_1_n_0 ;
  wire \i2c_data_i1[2]_i_1_n_0 ;
  wire \i2c_data_i1[3]_i_1_n_0 ;
  wire \i2c_data_i1[4]_i_1_n_0 ;
  wire \i2c_data_i1[5]_i_1_n_0 ;
  wire \i2c_data_i1[6]_i_1_n_0 ;
  wire \i2c_data_i1[7]_i_1__0_n_0 ;
  wire \i2c_data_i1[7]_i_1_n_0 ;
  wire [7:0]i2c_data_i2;
  wire \i2c_data_i2[7]_i_1__0_n_0 ;
  wire \i2c_data_i2[7]_i_1_n_0 ;
  wire [7:0]i2c_data_o0;
  wire [7:0]i2c_data_o1;
  wire i2c_error;
  wire [2:0]i2c_prog_cache_addr;
  wire \i2c_prog_cache_addr[0]_i_1_n_0 ;
  wire \i2c_prog_cache_addr[1]_i_1_n_0 ;
  wire \i2c_prog_cache_addr[2]_i_1_n_0 ;
  wire id_changed;
  wire id_changed0;
  wire id_changed_i_2_n_0;
  wire id_valid;
  wire [4:0]idovr;
  wire idovr0;
  wire \idovr[4]_i_2_n_0 ;
  wire \jtag_ctrl_reg_n_0_[0] ;
  wire \jtag_ctrl_reg_n_0_[1] ;
  wire last_reg_i_1_n_0;
  wire last_wr_cache_reg_0_7_0_0_i_2_n_0;
  wire led_dnlink_on;
  wire led_error;
  wire led_good;
  wire led_uplink_on;
  wire [8:0]mem_read_data_reg0;
  wire [8:0]mem_read_data_reg0__0;
  wire \mem_read_data_reg[8]_i_1__0_n_0 ;
  wire \mem_read_data_reg[8]_i_1_n_0 ;
  wire \mem_read_data_reg[8]_i_2__0_n_0 ;
  wire \mem_read_data_reg[8]_i_2_n_0 ;
  wire mem_read_data_valid_reg_i_1__0_n_0;
  wire mem_read_data_valid_reg_i_1_n_0;
  wire mem_reg_0_3_0_5_i_10_n_0;
  wire mem_reg_0_3_0_5_i_11_n_0;
  wire mem_reg_0_3_0_5_i_12_n_0;
  wire mem_reg_0_3_0_5_i_13_n_0;
  wire mem_reg_0_3_0_5_i_14_n_0;
  wire mem_reg_0_3_0_5_i_15_n_0;
  wire mem_reg_0_3_0_5_i_1__0_n_0;
  wire mem_reg_0_3_0_5_i_2_n_0;
  wire mem_reg_0_3_0_5_i_3_n_0;
  wire mem_reg_0_3_0_5_i_4_n_0;
  wire mem_reg_0_3_0_5_i_5_n_0;
  wire mem_reg_0_3_0_5_i_6_n_0;
  wire mem_reg_0_3_0_5_i_7_n_0;
  wire mem_reg_0_3_0_5_i_8_n_0;
  wire mem_reg_0_3_0_5_i_9_n_0;
  wire mem_reg_0_3_6_8_i_1_n_0;
  wire mem_reg_0_3_6_8_i_2_n_0;
  wire mem_reg_0_3_6_8_i_4_n_0;
  wire mem_reg_0_3_6_8_i_5_n_0;
  wire mem_reg_0_3_6_8_i_6_n_0;
  wire mem_reg_0_3_6_8_i_7_n_0;
  wire missed_ack_reg_i_1_n_0;
  wire mode_ping_reg_i_1_n_0;
  wire mode_stop_reg_i_2_n_0;
  wire mode_stop_reg_i_3_n_0;
  wire \out[0]_i_1_n_0 ;
  wire \out[7]_i_1_n_0 ;
  wire \output_axis_reg[8]_i_1__0_n_0 ;
  wire \output_axis_reg[8]_i_1_n_0 ;
  wire output_axis_tvalid_reg_i_1__0_n_0;
  wire output_axis_tvalid_reg_i_1_n_0;
  wire [15:0]p_0_in;
  wire p_0_in0_in;
  wire p_0_in__0;
  wire [6:0]p_0_in__1;
  wire [7:0]p_0_in__2;
  wire [3:0]p_0_in__3;
  wire p_1_in;
  wire [15:0]p_1_in__0;
  wire [15:8]p_29_in;
  wire p_2_in;
  wire phy_rx_data_reg_i_1_n_0;
  wire phy_rx_data_reg_i_2_n_0;
  wire \prog_cache_entries[0]_i_1_n_0 ;
  wire \prog_cache_entries[1]_i_1_n_0 ;
  wire \prog_cache_entries[2]_i_1_n_0 ;
  wire \prog_cache_entries[3]_i_1_n_0 ;
  wire prog_cache_reg_0_7_0_5_i_1_n_0;
  wire prog_cache_reg_0_7_0_5_i_2_n_0;
  wire prog_cache_reg_0_7_0_5_i_3_n_0;
  wire prog_cache_reg_0_7_0_5_i_4_n_0;
  wire prog_cache_reg_0_7_12_15_i_1_n_0;
  wire prog_jdi;
  wire prog_jen;
  wire \r_inbus[0]_i_1_n_0 ;
  wire \r_inbus[10]_i_1_n_0 ;
  wire \r_inbus[11]_i_1_n_0 ;
  wire \r_inbus[12]_i_1_n_0 ;
  wire \r_inbus[13]_i_1_n_0 ;
  wire \r_inbus[14]_i_1_n_0 ;
  wire \r_inbus[15]_i_1_n_0 ;
  wire \r_inbus[16]_i_1_n_0 ;
  wire \r_inbus[17]_i_1_n_0 ;
  wire \r_inbus[18]_i_1_n_0 ;
  wire \r_inbus[19]_i_1_n_0 ;
  wire \r_inbus[1]_i_1_n_0 ;
  wire \r_inbus[20]_i_1_n_0 ;
  wire \r_inbus[21]_i_1_n_0 ;
  wire \r_inbus[22]_i_1_n_0 ;
  wire \r_inbus[23]_i_1_n_0 ;
  wire \r_inbus[24]_i_1_n_0 ;
  wire \r_inbus[25]_i_1_n_0 ;
  wire \r_inbus[26]_i_1_n_0 ;
  wire \r_inbus[27]_i_1_n_0 ;
  wire \r_inbus[28]_i_1_n_0 ;
  wire \r_inbus[29]_i_1_n_0 ;
  wire \r_inbus[2]_i_1_n_0 ;
  wire \r_inbus[30]_i_1_n_0 ;
  wire \r_inbus[31]_i_1_n_0 ;
  wire \r_inbus[3]_i_1_n_0 ;
  wire \r_inbus[4]_i_1_n_0 ;
  wire \r_inbus[5]_i_1_n_0 ;
  wire \r_inbus[6]_i_1_n_0 ;
  wire \r_inbus[7]_i_1_n_0 ;
  wire \r_inbus[8]_i_1_n_0 ;
  wire \r_inbus[9]_i_1_n_0 ;
  wire \rd_addr_reg[0]_i_1__0_n_0 ;
  wire \rd_addr_reg[0]_i_1_n_0 ;
  wire \rd_addr_reg[1]_i_1__0_n_0 ;
  wire \rd_addr_reg[1]_i_1_n_0 ;
  wire rdy;
  wire [3:0]rfmod_id;
  wire \rfmod_id[0]_i_1_n_0 ;
  wire \rfmod_id[1]_i_1_n_0 ;
  wire \rfmod_id[2]_i_1_n_0 ;
  wire \rfmod_id[3]_i_1_n_0 ;
  wire \rfmod_id[3]_i_2_n_0 ;
  wire [16:1]rfmod_in;
  wire [16:1]rfmod_oe;
  wire \rfmod_oe[13]_INST_0_i_1_n_0 ;
  wire \rfmod_oe[13]_INST_0_i_2_n_0 ;
  wire \rfmod_oe[16]_INST_0_i_1_n_0 ;
  wire \rfmod_oe[8]_INST_0_i_1_n_0 ;
  wire [16:1]rfmod_out;
  wire \rfmod_out[11]_INST_0_i_1_n_0 ;
  wire \rfmod_out[12]_INST_0_i_1_n_0 ;
  wire \rfmod_out[12]_INST_0_i_2_n_0 ;
  wire \rfmod_out[13]_INST_0_i_1_n_0 ;
  wire \rfmod_out[13]_INST_0_i_2_n_0 ;
  wire \rfmod_out[13]_INST_0_i_3_n_0 ;
  wire \rfmod_out[14]_INST_0_i_1_n_0 ;
  wire \rfmod_out[14]_INST_0_i_2_n_0 ;
  wire \rfmod_out[14]_INST_0_i_3_n_0 ;
  wire \rfmod_out[15]_INST_0_i_1_n_0 ;
  wire \rfmod_out[16]_INST_0_i_1_n_0 ;
  wire \rfmod_out[16]_INST_0_i_2_n_0 ;
  wire \rfmod_out[5]_INST_0_i_1_n_0 ;
  wire \rfmod_out[6]_INST_0_i_1_n_0 ;
  wire \rfmod_out[6]_INST_0_i_2_n_0 ;
  wire \rfmod_out[7]_INST_0_i_1_n_0 ;
  wire \rfmod_out[7]_INST_0_i_2_n_0 ;
  wire \rfmod_out[8]_INST_0_i_1_n_0 ;
  wire \rfmod_out[8]_INST_0_i_2_n_0 ;
  wire \rfmod_out[9]_INST_0_i_1_n_0 ;
  wire \rfmod_out[9]_INST_0_i_2_n_0 ;
  wire \rfmod_out[9]_INST_0_i_3_n_0 ;
  wire rst;
  wire \rx_err_cnt[7]_i_3_n_0 ;
  wire rx_err_s_i_1_n_0;
  wire rx_hisel;
  wire scl_o_reg_i_1_n_0;
  wire scl_t;
  wire [15:0]scratch;
  wire \scratch[15]_i_1_n_0 ;
  wire \scratch[15]_i_2_n_0 ;
  wire sda_o_reg_i_10_n_0;
  wire sda_o_reg_i_11_n_0;
  wire sda_o_reg_i_12_n_0;
  wire sda_o_reg_i_13_n_0;
  wire sda_o_reg_i_14_n_0;
  wire sda_o_reg_i_15_n_0;
  wire sda_o_reg_i_16_n_0;
  wire sda_o_reg_i_17_n_0;
  wire sda_o_reg_i_1_n_0;
  wire sda_o_reg_i_2_n_0;
  wire sda_o_reg_i_3_n_0;
  wire sda_o_reg_i_4_n_0;
  wire sda_o_reg_i_5_n_0;
  wire sda_o_reg_i_6_n_0;
  wire sda_o_reg_i_7_n_0;
  wire sda_o_reg_i_8_n_0;
  wire sda_o_reg_i_9_n_0;
  wire sda_t;
  wire \slip_cnt[3]_i_2_n_0 ;
  wire \spi_bit_cnt[0]_i_1_n_0 ;
  wire \spi_bit_cnt[1]_i_1_n_0 ;
  wire \spi_bit_cnt[2]_i_1_n_0 ;
  wire \spi_bit_cnt[3]_i_1_n_0 ;
  wire \spi_bit_cnt[4]_i_2_n_0 ;
  wire \spi_bit_div[0]_i_1_n_0 ;
  wire \spi_bit_div[1]_i_1_n_0 ;
  wire \spi_bit_div[2]_i_1_n_0 ;
  wire \spi_bit_div[3]_i_1_n_0 ;
  wire \spi_bit_div[4]_i_1_n_0 ;
  wire \spi_bit_div[5]_i_1_n_0 ;
  wire \spi_bit_div[5]_i_2_n_0 ;
  wire \spi_bit_div[6]_i_2_n_0 ;
  wire \spi_bit_div[6]_i_3_n_0 ;
  wire \spi_bit_div[6]_i_4_n_0 ;
  wire spi_cs_n;
  wire spi_cs_n_i_1_n_0;
  wire spi_mosi;
  wire spi_mosi_i_1_n_0;
  wire spi_mosi_i_2_n_0;
  wire spi_sclk;
  wire spi_sclk_i_1_n_0;
  wire spi_sclk_i_2_n_0;
  wire \st_addr[0]_i_1_n_0 ;
  wire \st_addr[1]_i_1_n_0 ;
  wire \st_addr[2]_i_1_n_0 ;
  wire \st_cur[0]_i_10_n_0 ;
  wire \st_cur[0]_i_11_n_0 ;
  wire \st_cur[0]_i_12_n_0 ;
  wire \st_cur[0]_i_13_n_0 ;
  wire \st_cur[0]_i_1_n_0 ;
  wire \st_cur[0]_i_2_n_0 ;
  wire \st_cur[0]_i_3_n_0 ;
  wire \st_cur[0]_i_4_n_0 ;
  wire \st_cur[0]_i_5_n_0 ;
  wire \st_cur[0]_i_6_n_0 ;
  wire \st_cur[0]_i_7_n_0 ;
  wire \st_cur[0]_i_8_n_0 ;
  wire \st_cur[0]_i_9_n_0 ;
  wire \st_cur[1]_i_1_n_0 ;
  wire \st_cur[1]_i_2_n_0 ;
  wire \st_cur[1]_i_3_n_0 ;
  wire \st_cur[1]_i_4_n_0 ;
  wire \state[0]_i_1_n_0 ;
  wire \state[1]_i_1_n_0 ;
  wire \state[1]_i_2_n_0 ;
  wire \state[2]_i_1_n_0 ;
  wire \state[2]_i_2_n_0 ;
  wire \state[2]_i_3_n_0 ;
  wire \state[2]_i_4_n_0 ;
  wire sync;
  wire \test_ctrl[15]_i_2_n_0 ;
  wire test_dir;
  wire test_en;
  wire [29:21]test_stat;
  wire [19:0]test_stat__0;
  wire trx_auto;
  wire [1:0]tx_active;
  wire \tx_cnt[0]_i_1_n_0 ;
  wire \tx_cnt[0]_i_3_n_0 ;
  wire \tx_cnt_reg[0]_i_2_n_0 ;
  wire \tx_cnt_reg[0]_i_2_n_1 ;
  wire \tx_cnt_reg[0]_i_2_n_2 ;
  wire \tx_cnt_reg[0]_i_2_n_3 ;
  wire \tx_cnt_reg[0]_i_2_n_4 ;
  wire \tx_cnt_reg[0]_i_2_n_5 ;
  wire \tx_cnt_reg[0]_i_2_n_6 ;
  wire \tx_cnt_reg[0]_i_2_n_7 ;
  wire \tx_cnt_reg[12]_i_1_n_0 ;
  wire \tx_cnt_reg[12]_i_1_n_1 ;
  wire \tx_cnt_reg[12]_i_1_n_2 ;
  wire \tx_cnt_reg[12]_i_1_n_3 ;
  wire \tx_cnt_reg[12]_i_1_n_4 ;
  wire \tx_cnt_reg[12]_i_1_n_5 ;
  wire \tx_cnt_reg[12]_i_1_n_6 ;
  wire \tx_cnt_reg[12]_i_1_n_7 ;
  wire \tx_cnt_reg[16]_i_1_n_0 ;
  wire \tx_cnt_reg[16]_i_1_n_1 ;
  wire \tx_cnt_reg[16]_i_1_n_2 ;
  wire \tx_cnt_reg[16]_i_1_n_3 ;
  wire \tx_cnt_reg[16]_i_1_n_4 ;
  wire \tx_cnt_reg[16]_i_1_n_5 ;
  wire \tx_cnt_reg[16]_i_1_n_6 ;
  wire \tx_cnt_reg[16]_i_1_n_7 ;
  wire \tx_cnt_reg[20]_i_1_n_0 ;
  wire \tx_cnt_reg[20]_i_1_n_1 ;
  wire \tx_cnt_reg[20]_i_1_n_2 ;
  wire \tx_cnt_reg[20]_i_1_n_3 ;
  wire \tx_cnt_reg[20]_i_1_n_4 ;
  wire \tx_cnt_reg[20]_i_1_n_5 ;
  wire \tx_cnt_reg[20]_i_1_n_6 ;
  wire \tx_cnt_reg[20]_i_1_n_7 ;
  wire \tx_cnt_reg[24]_i_1_n_0 ;
  wire \tx_cnt_reg[24]_i_1_n_1 ;
  wire \tx_cnt_reg[24]_i_1_n_2 ;
  wire \tx_cnt_reg[24]_i_1_n_3 ;
  wire \tx_cnt_reg[24]_i_1_n_4 ;
  wire \tx_cnt_reg[24]_i_1_n_5 ;
  wire \tx_cnt_reg[24]_i_1_n_6 ;
  wire \tx_cnt_reg[24]_i_1_n_7 ;
  wire \tx_cnt_reg[28]_i_1_n_2 ;
  wire \tx_cnt_reg[28]_i_1_n_3 ;
  wire \tx_cnt_reg[28]_i_1_n_5 ;
  wire \tx_cnt_reg[28]_i_1_n_6 ;
  wire \tx_cnt_reg[28]_i_1_n_7 ;
  wire \tx_cnt_reg[4]_i_1_n_0 ;
  wire \tx_cnt_reg[4]_i_1_n_1 ;
  wire \tx_cnt_reg[4]_i_1_n_2 ;
  wire \tx_cnt_reg[4]_i_1_n_3 ;
  wire \tx_cnt_reg[4]_i_1_n_4 ;
  wire \tx_cnt_reg[4]_i_1_n_5 ;
  wire \tx_cnt_reg[4]_i_1_n_6 ;
  wire \tx_cnt_reg[4]_i_1_n_7 ;
  wire \tx_cnt_reg[8]_i_1_n_0 ;
  wire \tx_cnt_reg[8]_i_1_n_1 ;
  wire \tx_cnt_reg[8]_i_1_n_2 ;
  wire \tx_cnt_reg[8]_i_1_n_3 ;
  wire \tx_cnt_reg[8]_i_1_n_4 ;
  wire \tx_cnt_reg[8]_i_1_n_5 ;
  wire \tx_cnt_reg[8]_i_1_n_6 ;
  wire \tx_cnt_reg[8]_i_1_n_7 ;
  wire tx_hisel;
  wire [6:0]\u_i2c_master/addr_reg ;
  wire [3:0]\u_i2c_master/bit_count_reg ;
  wire \u_i2c_master/data_in_last ;
  wire \u_i2c_master/data_out_last_reg ;
  wire [7:0]\u_i2c_master/data_out_reg ;
  wire [16:1]\u_i2c_master/delay_next0 ;
  wire [16:0]\u_i2c_master/delay_reg ;
  wire \u_i2c_master/ext_cmd_queued0 ;
  wire [6:0]\u_i2c_master/i2c_addr_s ;
  wire [3:0]\u_i2c_master/i2c_cmd_mode ;
  wire \u_i2c_master/i2c_cmd_mode_reg_n_0_[0] ;
  wire \u_i2c_master/i2c_cmd_mode_reg_n_0_[1] ;
  wire \u_i2c_master/i2c_cmd_mode_reg_n_0_[2] ;
  wire \u_i2c_master/i2c_cmd_mode_reg_n_0_[3] ;
  wire [7:0]\u_i2c_master/i2c_data_i0 ;
  wire [7:0]\u_i2c_master/i2c_data_i1 ;
  wire [7:0]\u_i2c_master/i2c_data_i2 ;
  wire [8:0]\u_i2c_master/mem_read_data_reg ;
  wire [3:0]\u_i2c_master/prog_cache_entries ;
  wire \u_i2c_master/prog_cache_reg_0_7_0_5_n_0 ;
  wire \u_i2c_master/prog_cache_reg_0_7_0_5_n_1 ;
  wire \u_i2c_master/prog_cache_reg_0_7_0_5_n_2 ;
  wire \u_i2c_master/prog_cache_reg_0_7_0_5_n_3 ;
  wire \u_i2c_master/prog_cache_reg_0_7_0_5_n_4 ;
  wire \u_i2c_master/prog_cache_reg_0_7_0_5_n_5 ;
  wire \u_i2c_master/prog_cache_reg_0_7_12_15_n_0 ;
  wire \u_i2c_master/prog_cache_reg_0_7_12_15_n_1 ;
  wire \u_i2c_master/prog_cache_reg_0_7_12_15_n_3 ;
  wire \u_i2c_master/prog_cache_reg_0_7_6_11_n_0 ;
  wire \u_i2c_master/prog_cache_reg_0_7_6_11_n_1 ;
  wire \u_i2c_master/prog_cache_reg_0_7_6_11_n_2 ;
  wire \u_i2c_master/prog_cache_reg_0_7_6_11_n_3 ;
  wire \u_i2c_master/prog_cache_reg_0_7_6_11_n_4 ;
  wire \u_i2c_master/prog_cache_reg_0_7_6_11_n_5 ;
  wire [2:2]\u_i2c_master/rd_ptr_next ;
  wire [2:2]\u_i2c_master/rd_ptr_next__0 ;
  wire [2:0]\u_i2c_master/rd_ptr_reg ;
  wire \u_i2c_master/st_addr_reg_n_0_[0] ;
  wire \u_i2c_master/st_addr_reg_n_0_[1] ;
  wire \u_i2c_master/st_addr_reg_n_0_[2] ;
  wire [7:0]\u_i2c_master/st_is_dirty0 ;
  wire [7:0]\u_i2c_master/st_reg_data ;
  wire \u_i2c_master/st_wr_mode ;
  wire \u_i2c_master/state_reg_n_0_[0] ;
  wire \u_i2c_master/state_reg_n_0_[1] ;
  wire \u_i2c_master/state_reg_n_0_[2] ;
  wire \u_i2c_master/u_i2c_master/cmd_mode_r ;
  wire \u_i2c_master/u_i2c_master/cmd_mode_r_reg_n_0_[0] ;
  wire \u_i2c_master/u_i2c_master/cmd_ping09_out ;
  wire \u_i2c_master/u_i2c_master/cmd_read0 ;
  wire \u_i2c_master/u_i2c_master/cmd_ready0 ;
  wire \u_i2c_master/u_i2c_master/cmd_stop0 ;
  wire \u_i2c_master/u_i2c_master/cmd_write_multiple0 ;
  wire \u_i2c_master/u_i2c_master/data_in_last ;
  wire \u_i2c_master/u_i2c_master/data_in_ready0 ;
  wire \u_i2c_master/u_i2c_master/data_in_valid0 ;
  wire \u_i2c_master/u_i2c_master/data_o0_reg_n_0_[0] ;
  wire \u_i2c_master/u_i2c_master/data_o0_reg_n_0_[1] ;
  wire \u_i2c_master/u_i2c_master/data_o0_reg_n_0_[2] ;
  wire \u_i2c_master/u_i2c_master/data_o0_reg_n_0_[3] ;
  wire \u_i2c_master/u_i2c_master/data_o0_reg_n_0_[4] ;
  wire \u_i2c_master/u_i2c_master/data_o0_reg_n_0_[5] ;
  wire \u_i2c_master/u_i2c_master/data_o0_reg_n_0_[6] ;
  wire \u_i2c_master/u_i2c_master/data_o0_reg_n_0_[7] ;
  wire \u_i2c_master/u_i2c_master/data_o1_reg_n_0_[0] ;
  wire \u_i2c_master/u_i2c_master/data_o1_reg_n_0_[1] ;
  wire \u_i2c_master/u_i2c_master/data_o1_reg_n_0_[2] ;
  wire \u_i2c_master/u_i2c_master/data_o1_reg_n_0_[3] ;
  wire \u_i2c_master/u_i2c_master/data_o1_reg_n_0_[4] ;
  wire \u_i2c_master/u_i2c_master/data_o1_reg_n_0_[5] ;
  wire \u_i2c_master/u_i2c_master/data_o1_reg_n_0_[6] ;
  wire \u_i2c_master/u_i2c_master/data_o1_reg_n_0_[7] ;
  wire \u_i2c_master/u_i2c_master/data_out_last ;
  wire \u_i2c_master/u_i2c_master/data_out_valid ;
  wire \u_i2c_master/u_i2c_master/data_out_valid0 ;
  wire \u_i2c_master/u_i2c_master/error_reg_n_0 ;
  wire \u_i2c_master/u_i2c_master/p_0_in7_in ;
  wire \u_i2c_master/u_i2c_master/p_1_in ;
  wire \u_i2c_master/u_i2c_master/p_4_in ;
  wire [3:0]\u_i2c_master/u_i2c_master/state ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[0] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[10] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[11] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[1] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[3] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[6] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[7] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ;
  wire [3:0]\u_i2c_master/u_i2c_master/u_i2c_master/bit_count_next ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/bus_active_reg_reg_n_0 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/busy_reg0 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/busy_reg_reg_n_0 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/data_in_ready_next ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/data_reg_reg_n_0_[7] ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__0_n_0 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__0_n_1 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__0_n_2 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__0_n_3 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__1_n_0 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__1_n_1 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__1_n_2 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__1_n_3 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__2_n_1 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__2_n_2 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__2_n_3 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry_n_0 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry_n_1 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry_n_2 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry_n_3 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/delay_scl_reg ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/last_reg ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/last_sda_i_reg ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/missed_ack_reg ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/missed_ack_reg_reg_n_0 ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/mode_ping_reg ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/mode_read_reg ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/mode_write_multiple_reg ;
  wire [7:0]\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/phy_state_next ;
  wire [3:0]\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/scl_i_reg ;
  wire \u_i2c_master/u_i2c_master/u_i2c_master/sda_i_reg ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[0] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[1] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[2] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[3] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[4] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[5] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[6] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[7] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[8] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_valid_reg ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[0] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[1] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[2] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[3] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[4] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[5] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[6] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[7] ;
  wire [1:0]\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_addr_reg ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[0] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[1] ;
  wire \u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[2] ;
  wire [1:0]\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_addr_reg ;
  wire [1:0]\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_next ;
  wire [2:0]\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg ;
  wire \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_valid_reg ;
  wire \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[0] ;
  wire \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[1] ;
  wire \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[2] ;
  wire \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[3] ;
  wire \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[4] ;
  wire \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[5] ;
  wire \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[6] ;
  wire \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[7] ;
  wire [1:0]\u_i2c_master/u_i2c_master/u_wdata_fifo/rd_addr_reg ;
  wire [1:0]\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_addr_reg ;
  wire [1:0]\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_next ;
  wire [2:0]\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg ;
  wire \u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ;
  wire \u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[1] ;
  wire \u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ;
  wire \u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[3] ;
  wire \u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ;
  wire \u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[5] ;
  wire \u_spi_gpio_exp_master/dat_cache ;
  wire \u_spi_gpio_exp_master/dat_cache1_carry__0_n_0 ;
  wire \u_spi_gpio_exp_master/dat_cache1_carry__0_n_1 ;
  wire \u_spi_gpio_exp_master/dat_cache1_carry__0_n_2 ;
  wire \u_spi_gpio_exp_master/dat_cache1_carry__0_n_3 ;
  wire \u_spi_gpio_exp_master/dat_cache1_carry__1_n_1 ;
  wire \u_spi_gpio_exp_master/dat_cache1_carry__1_n_2 ;
  wire \u_spi_gpio_exp_master/dat_cache1_carry__1_n_3 ;
  wire \u_spi_gpio_exp_master/dat_cache1_carry_n_0 ;
  wire \u_spi_gpio_exp_master/dat_cache1_carry_n_1 ;
  wire \u_spi_gpio_exp_master/dat_cache1_carry_n_2 ;
  wire \u_spi_gpio_exp_master/dat_cache1_carry_n_3 ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[0] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[10] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[11] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[12] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[13] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[14] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[15] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[16] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[17] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[18] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[19] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[1] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[20] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[21] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[22] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[23] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[24] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[25] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[26] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[27] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[28] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[29] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[2] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[30] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[31] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[3] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[4] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[5] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[6] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[7] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[8] ;
  wire \u_spi_gpio_exp_master/dat_cache_reg_n_0_[9] ;
  wire [31:1]\u_spi_gpio_exp_master/in15 ;
  wire [31:1]\u_spi_gpio_exp_master/in17 ;
  wire \u_spi_gpio_exp_master/p_0_in ;
  wire \u_spi_gpio_exp_master/r_inbus ;
  wire \u_spi_gpio_exp_master/r_inbus_reg_n_0_[31] ;
  wire \u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[0] ;
  wire \u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[1] ;
  wire \u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[2] ;
  wire \u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[3] ;
  wire \u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[4] ;
  wire \u_spi_gpio_exp_master/spi_bit_div ;
  wire \u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[0] ;
  wire \u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[1] ;
  wire \u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[2] ;
  wire \u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[3] ;
  wire \u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[4] ;
  wire \u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[5] ;
  wire \u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[6] ;
  wire \u_spi_gpio_exp_master/w_outbus ;
  wire \u_tester/chk_en ;
  wire [7:0]\u_tester/chk_out ;
  wire [6:0]\u_tester/count_reg ;
  wire \u_tester/mask_count ;
  wire \u_tester/mask_count_reg_n_0_[0] ;
  wire \u_tester/mask_count_reg_n_0_[1] ;
  wire \u_tester/mask_count_reg_n_0_[2] ;
  wire \u_tester/mask_count_reg_n_0_[3] ;
  wire \u_tester/mask_count_reg_n_0_[4] ;
  wire \u_tester/mask_count_reg_n_0_[5] ;
  wire \u_tester/mask_count_reg_n_0_[6] ;
  wire \u_tester/mask_count_reg_n_0_[7] ;
  wire [0:0]\u_tester/p_0_out ;
  wire \u_tester/rx_err_cnt0 ;
  wire \u_tester/slip_cnt0 ;
  wire [1:0]\u_tester/st_cur ;
  wire \u_tester/tx_cnt_reg_n_0_[0] ;
  wire \u_tester/tx_cnt_reg_n_0_[10] ;
  wire \u_tester/tx_cnt_reg_n_0_[11] ;
  wire \u_tester/tx_cnt_reg_n_0_[12] ;
  wire \u_tester/tx_cnt_reg_n_0_[13] ;
  wire \u_tester/tx_cnt_reg_n_0_[14] ;
  wire \u_tester/tx_cnt_reg_n_0_[15] ;
  wire \u_tester/tx_cnt_reg_n_0_[16] ;
  wire \u_tester/tx_cnt_reg_n_0_[17] ;
  wire \u_tester/tx_cnt_reg_n_0_[18] ;
  wire \u_tester/tx_cnt_reg_n_0_[19] ;
  wire \u_tester/tx_cnt_reg_n_0_[1] ;
  wire \u_tester/tx_cnt_reg_n_0_[20] ;
  wire \u_tester/tx_cnt_reg_n_0_[21] ;
  wire \u_tester/tx_cnt_reg_n_0_[22] ;
  wire \u_tester/tx_cnt_reg_n_0_[2] ;
  wire \u_tester/tx_cnt_reg_n_0_[3] ;
  wire \u_tester/tx_cnt_reg_n_0_[4] ;
  wire \u_tester/tx_cnt_reg_n_0_[5] ;
  wire \u_tester/tx_cnt_reg_n_0_[6] ;
  wire \u_tester/tx_cnt_reg_n_0_[7] ;
  wire \u_tester/tx_cnt_reg_n_0_[8] ;
  wire \u_tester/tx_cnt_reg_n_0_[9] ;
  wire \u_tester/u_gen/out_reg_n_0_[0] ;
  wire \u_tester/u_gen/out_reg_n_0_[1] ;
  wire \u_tester/u_gen/out_reg_n_0_[2] ;
  wire \u_tester/u_gen/out_reg_n_0_[3] ;
  wire \u_tester/u_gen/out_reg_n_0_[4] ;
  wire \u_tester/u_gen/out_reg_n_0_[5] ;
  wire \u_tester/u_gen/out_reg_n_0_[6] ;
  wire \u_tester/u_gen/out_reg_n_0_[7] ;
  wire \w_outbus[0]_i_1_n_0 ;
  wire \w_outbus[10]_i_1_n_0 ;
  wire \w_outbus[10]_i_2_n_0 ;
  wire \w_outbus[11]_i_1_n_0 ;
  wire \w_outbus[11]_i_2_n_0 ;
  wire \w_outbus[12]_i_1_n_0 ;
  wire \w_outbus[12]_i_2_n_0 ;
  wire \w_outbus[13]_i_1_n_0 ;
  wire \w_outbus[13]_i_2_n_0 ;
  wire \w_outbus[14]_i_1_n_0 ;
  wire \w_outbus[14]_i_2_n_0 ;
  wire \w_outbus[15]_i_1_n_0 ;
  wire \w_outbus[15]_i_2_n_0 ;
  wire \w_outbus[16]_i_1_n_0 ;
  wire \w_outbus[16]_i_2_n_0 ;
  wire \w_outbus[17]_i_1_n_0 ;
  wire \w_outbus[17]_i_2_n_0 ;
  wire \w_outbus[18]_i_1_n_0 ;
  wire \w_outbus[18]_i_2_n_0 ;
  wire \w_outbus[19]_i_1_n_0 ;
  wire \w_outbus[19]_i_2_n_0 ;
  wire \w_outbus[1]_i_1_n_0 ;
  wire \w_outbus[1]_i_2_n_0 ;
  wire \w_outbus[20]_i_1_n_0 ;
  wire \w_outbus[20]_i_2_n_0 ;
  wire \w_outbus[21]_i_1_n_0 ;
  wire \w_outbus[21]_i_2_n_0 ;
  wire \w_outbus[22]_i_1_n_0 ;
  wire \w_outbus[22]_i_2_n_0 ;
  wire \w_outbus[23]_i_1_n_0 ;
  wire \w_outbus[23]_i_2_n_0 ;
  wire \w_outbus[24]_i_1_n_0 ;
  wire \w_outbus[24]_i_2_n_0 ;
  wire \w_outbus[25]_i_1_n_0 ;
  wire \w_outbus[25]_i_2_n_0 ;
  wire \w_outbus[26]_i_1_n_0 ;
  wire \w_outbus[26]_i_2_n_0 ;
  wire \w_outbus[27]_i_1_n_0 ;
  wire \w_outbus[27]_i_2_n_0 ;
  wire \w_outbus[28]_i_1_n_0 ;
  wire \w_outbus[29]_i_1_n_0 ;
  wire \w_outbus[2]_i_1_n_0 ;
  wire \w_outbus[2]_i_2_n_0 ;
  wire \w_outbus[30]_i_1_n_0 ;
  wire \w_outbus[31]_i_2_n_0 ;
  wire \w_outbus[31]_i_3_n_0 ;
  wire \w_outbus[31]_i_4_n_0 ;
  wire \w_outbus[3]_i_1_n_0 ;
  wire \w_outbus[3]_i_2_n_0 ;
  wire \w_outbus[4]_i_1_n_0 ;
  wire \w_outbus[5]_i_1_n_0 ;
  wire \w_outbus[6]_i_1_n_0 ;
  wire \w_outbus[7]_i_1_n_0 ;
  wire \w_outbus[8]_i_1_n_0 ;
  wire \w_outbus[9]_i_1_n_0 ;
  wire wr;
  wire \wr_addr_reg[1]_i_2_n_0 ;
  wire wr_cache_reg_0_7_0_5_i_1_n_0;
  wire wr_cache_reg_0_7_0_5_i_2_n_0;
  wire wr_cache_reg_0_7_0_5_i_3_n_0;
  wire wr_cache_reg_0_7_0_5_i_4_n_0;
  wire wr_cache_reg_0_7_0_5_i_5_n_0;
  wire wr_cache_reg_0_7_0_5_i_6_n_0;
  wire wr_cache_reg_0_7_0_5_i_7_n_0;
  wire wr_cache_reg_0_7_0_5_i_8_n_0;
  wire wr_cache_reg_0_7_6_7_i_1_n_0;
  wire wr_cache_reg_0_7_6_7_i_2_n_0;
  wire wr_cache_reg_0_7_6_7_i_3_n_0;
  wire [2:2]wr_ptr_next;
  wire [2:2]wr_ptr_next__0;
  wire write1_out__0;
  wire [3:2]\NLW_tx_cnt_reg[28]_i_1_CO_UNCONNECTED ;
  wire [3:3]\NLW_tx_cnt_reg[28]_i_1_O_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/prog_cache_reg_0_7_0_5_DOD_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/prog_cache_reg_0_7_12_15_DOC_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/prog_cache_reg_0_7_12_15_DOD_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/prog_cache_reg_0_7_6_11_DOD_UNCONNECTED ;
  wire [3:3]\NLW_u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__2_CO_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/u_i2c_master/u_rdata_fifo/mem_reg_0_3_0_5_DOD_UNCONNECTED ;
  wire [1:1]\NLW_u_i2c_master/u_i2c_master/u_rdata_fifo/mem_reg_0_3_6_8_DOB_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/u_i2c_master/u_rdata_fifo/mem_reg_0_3_6_8_DOC_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/u_i2c_master/u_rdata_fifo/mem_reg_0_3_6_8_DOD_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/u_i2c_master/u_wdata_fifo/mem_reg_0_3_0_5_DOD_UNCONNECTED ;
  wire [1:1]\NLW_u_i2c_master/u_i2c_master/u_wdata_fifo/mem_reg_0_3_6_8_DOB_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/u_i2c_master/u_wdata_fifo/mem_reg_0_3_6_8_DOC_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/u_i2c_master/u_wdata_fifo/mem_reg_0_3_6_8_DOD_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/wr_cache_reg_0_7_0_5_DOD_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/wr_cache_reg_0_7_6_7_DOB_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/wr_cache_reg_0_7_6_7_DOC_UNCONNECTED ;
  wire [1:0]\NLW_u_i2c_master/wr_cache_reg_0_7_6_7_DOD_UNCONNECTED ;
  wire [3:0]\NLW_u_spi_gpio_exp_master/dat_cache1_carry_O_UNCONNECTED ;
  wire [3:0]\NLW_u_spi_gpio_exp_master/dat_cache1_carry__0_O_UNCONNECTED ;
  wire [3:3]\NLW_u_spi_gpio_exp_master/dat_cache1_carry__1_CO_UNCONNECTED ;
  wire [3:0]\NLW_u_spi_gpio_exp_master/dat_cache1_carry__1_O_UNCONNECTED ;

  assign cbrs_rev[3] = \<const0> ;
  assign cbrs_rev[2] = \<const0> ;
  assign cbrs_rev[1] = \<const0> ;
  assign cbrs_rev[0] = \^cbrs_rev [0];
  assign debug[7] = \<const0> ;
  assign debug[6] = \<const0> ;
  assign debug[5] = \<const0> ;
  assign debug[4] = \<const0> ;
  assign debug[3] = \<const0> ;
  assign debug[2] = \<const0> ;
  assign debug[1] = \<const0> ;
  assign debug[0] = \<const0> ;
  LUT3 #(
    .INIT(8'h47)) 
    \FSM_onehot_state[0]_i_1 
       (.I0(gpio_exp_busy),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[5] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_div ),
        .O(\FSM_onehot_state[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFF4FF444FF4FFFFF)) 
    \FSM_onehot_state[5]_i_1 
       (.I0(\FSM_onehot_state[5]_i_2_n_0 ),
        .I1(\FSM_onehot_state[5]_i_3_n_0 ),
        .I2(gpio_exp_busy),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I4(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[5] ),
        .I5(\u_spi_gpio_exp_master/spi_bit_div ),
        .O(\FSM_onehot_state[5]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0001010101010101)) 
    \FSM_onehot_state[5]_i_2 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[1] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[3] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[4] ),
        .I4(\FSM_onehot_state[5]_i_4_n_0 ),
        .I5(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .O(\FSM_onehot_state[5]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000004000)) 
    \FSM_onehot_state[5]_i_3 
       (.I0(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[4] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[6] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[5] ),
        .I3(\FSM_onehot_state[5]_i_5_n_0 ),
        .I4(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[3] ),
        .I5(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[2] ),
        .O(\FSM_onehot_state[5]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair88" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \FSM_onehot_state[5]_i_4 
       (.I0(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[0] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[1] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[3] ),
        .O(\FSM_onehot_state[5]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \FSM_onehot_state[5]_i_5 
       (.I0(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[1] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[0] ),
        .O(\FSM_onehot_state[5]_i_5_n_0 ));
  LUT2 #(
    .INIT(4'hB)) 
    \FSM_onehot_state_reg[0]_i_1 
       (.I0(\FSM_onehot_state_reg[0]_i_2_n_0 ),
        .I1(\FSM_onehot_state_reg[0]_i_3_n_0 ),
        .O(\FSM_onehot_state_reg[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAA8AAA82AAAAAAAA)) 
    \FSM_onehot_state_reg[0]_i_2 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[0] ),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [1]),
        .I4(\u_i2c_master/u_i2c_master/state [0]),
        .I5(\u_i2c_master/u_i2c_master/cmd_ready0 ),
        .O(\FSM_onehot_state_reg[0]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h000000000BBBBBBB)) 
    \FSM_onehot_state_reg[0]_i_3 
       (.I0(\FSM_onehot_state_reg[7]_i_3_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[11] ),
        .I2(\FSM_onehot_state_reg[0]_i_4_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[10] ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg ),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[7] ),
        .O(\FSM_onehot_state_reg[0]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT2 #(
    .INIT(4'hB)) 
    \FSM_onehot_state_reg[0]_i_4 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/last_reg ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/mode_write_multiple_reg ),
        .O(\FSM_onehot_state_reg[0]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT5 #(
    .INIT(32'h00000002)) 
    \FSM_onehot_state_reg[10]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .I1(\u_i2c_master/bit_count_reg [0]),
        .I2(\u_i2c_master/bit_count_reg [1]),
        .I3(\u_i2c_master/bit_count_reg [2]),
        .I4(\u_i2c_master/bit_count_reg [3]),
        .O(\FSM_onehot_state_reg[10]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'h1001)) 
    \FSM_onehot_state_reg[11]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .O(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ));
  LUT6 #(
    .INIT(64'h4F4444444F444F44)) 
    \FSM_onehot_state_reg[11]_i_2 
       (.I0(\FSM_onehot_state_reg[11]_i_3_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[11] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[10] ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/last_reg ),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/mode_write_multiple_reg ),
        .O(\FSM_onehot_state_reg[11]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT5 #(
    .INIT(32'h02020208)) 
    \FSM_onehot_state_reg[11]_i_3 
       (.I0(\u_i2c_master/u_i2c_master/cmd_ready0 ),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/u_i2c_master/state [1]),
        .I3(\u_i2c_master/u_i2c_master/state [3]),
        .I4(\u_i2c_master/u_i2c_master/state [0]),
        .O(\FSM_onehot_state_reg[11]_i_3_n_0 ));
  LUT3 #(
    .INIT(8'hA8)) 
    \FSM_onehot_state_reg[1]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/bus_active_reg_reg_n_0 ),
        .I1(\FSM_onehot_state_reg[1]_i_2_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[1] ),
        .O(\FSM_onehot_state_reg[1]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0406000000000000)) 
    \FSM_onehot_state_reg[1]_i_2 
       (.I0(\u_i2c_master/u_i2c_master/state [2]),
        .I1(\u_i2c_master/u_i2c_master/state [3]),
        .I2(\u_i2c_master/u_i2c_master/state [1]),
        .I3(\u_i2c_master/u_i2c_master/state [0]),
        .I4(\u_i2c_master/u_i2c_master/cmd_ready0 ),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[0] ),
        .O(\FSM_onehot_state_reg[1]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFE0000FFFFFFFF)) 
    \FSM_onehot_state_reg[2]_i_1 
       (.I0(\u_i2c_master/bit_count_reg [3]),
        .I1(\u_i2c_master/bit_count_reg [2]),
        .I2(\u_i2c_master/bit_count_reg [1]),
        .I3(\u_i2c_master/bit_count_reg [0]),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .I5(\FSM_onehot_state_reg[2]_i_2_n_0 ),
        .O(\FSM_onehot_state_reg[2]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h3031303100003031)) 
    \FSM_onehot_state_reg[2]_i_2 
       (.I0(\FSM_onehot_state_reg[1]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[6] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/bus_active_reg_reg_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[1] ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[11] ),
        .I5(\FSM_onehot_state_reg[6]_i_3_n_0 ),
        .O(\FSM_onehot_state_reg[2]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT5 #(
    .INIT(32'h00000002)) 
    \FSM_onehot_state_reg[3]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .I1(\u_i2c_master/bit_count_reg [0]),
        .I2(\u_i2c_master/bit_count_reg [1]),
        .I3(\u_i2c_master/bit_count_reg [2]),
        .I4(\u_i2c_master/bit_count_reg [3]),
        .O(\FSM_onehot_state_reg[3]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hEEEEEEEEEEEEEEEA)) 
    \FSM_onehot_state_reg[4]_i_1 
       (.I0(\FSM_onehot_state_reg[4]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .I2(\u_i2c_master/bit_count_reg [3]),
        .I3(\u_i2c_master/bit_count_reg [2]),
        .I4(\u_i2c_master/bit_count_reg [1]),
        .I5(\u_i2c_master/bit_count_reg [0]),
        .O(\FSM_onehot_state_reg[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT5 #(
    .INIT(32'hFF404040)) 
    \FSM_onehot_state_reg[4]_i_2 
       (.I0(\FSM_onehot_state_reg[6]_i_3_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I2(\FSM_onehot_state_reg[6]_i_2_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[3] ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/mode_read_reg ),
        .O(\FSM_onehot_state_reg[4]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'h44444F44)) 
    \FSM_onehot_state_reg[5]_i_1 
       (.I0(\FSM_onehot_state_reg[11]_i_3_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I2(\FSM_onehot_state_reg[9]_i_2_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg ),
        .O(\FSM_onehot_state_reg[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair2" *) 
  LUT3 #(
    .INIT(8'h04)) 
    \FSM_onehot_state_reg[6]_i_1 
       (.I0(\FSM_onehot_state_reg[6]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I2(\FSM_onehot_state_reg[6]_i_3_n_0 ),
        .O(\FSM_onehot_state_reg[6]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0001000000000001)) 
    \FSM_onehot_state_reg[6]_i_2 
       (.I0(\FSM_onehot_state_reg[6]_i_4_n_0 ),
        .I1(\FSM_onehot_state_reg[6]_i_5_n_0 ),
        .I2(\FSM_onehot_state_reg[6]_i_6_n_0 ),
        .I3(\FSM_onehot_state_reg[6]_i_7_n_0 ),
        .I4(\u_i2c_master/i2c_addr_s [6]),
        .I5(\u_i2c_master/addr_reg [6]),
        .O(\FSM_onehot_state_reg[6]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFEBFFEAFFFFFFFFF)) 
    \FSM_onehot_state_reg[6]_i_3 
       (.I0(\u_i2c_master/u_i2c_master/state [1]),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\FSM_onehot_state_reg[6]_i_8_n_0 ),
        .I5(\u_i2c_master/u_i2c_master/cmd_ready0 ),
        .O(\FSM_onehot_state_reg[6]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair59" *) 
  LUT4 #(
    .INIT(16'h0100)) 
    \FSM_onehot_state_reg[6]_i_4 
       (.I0(\u_i2c_master/u_i2c_master/state [0]),
        .I1(\u_i2c_master/u_i2c_master/state [3]),
        .I2(\u_i2c_master/u_i2c_master/state [1]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .O(\FSM_onehot_state_reg[6]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair13" *) 
  LUT4 #(
    .INIT(16'h0004)) 
    \FSM_onehot_state_reg[6]_i_5 
       (.I0(\u_i2c_master/u_i2c_master/state [2]),
        .I1(\u_i2c_master/u_i2c_master/state [3]),
        .I2(\u_i2c_master/u_i2c_master/state [0]),
        .I3(\u_i2c_master/u_i2c_master/state [1]),
        .O(\FSM_onehot_state_reg[6]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h6FF6FFFFFFFF6FF6)) 
    \FSM_onehot_state_reg[6]_i_6 
       (.I0(\u_i2c_master/addr_reg [3]),
        .I1(\u_i2c_master/i2c_addr_s [3]),
        .I2(\u_i2c_master/i2c_addr_s [4]),
        .I3(\u_i2c_master/addr_reg [4]),
        .I4(\u_i2c_master/i2c_addr_s [5]),
        .I5(\u_i2c_master/addr_reg [5]),
        .O(\FSM_onehot_state_reg[6]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'h6FF6FFFFFFFF6FF6)) 
    \FSM_onehot_state_reg[6]_i_7 
       (.I0(\u_i2c_master/addr_reg [0]),
        .I1(\u_i2c_master/i2c_addr_s [0]),
        .I2(\u_i2c_master/i2c_addr_s [1]),
        .I3(\u_i2c_master/addr_reg [1]),
        .I4(\u_i2c_master/i2c_addr_s [2]),
        .I5(\u_i2c_master/addr_reg [2]),
        .O(\FSM_onehot_state_reg[6]_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT4 #(
    .INIT(16'h0001)) 
    \FSM_onehot_state_reg[6]_i_8 
       (.I0(\u_i2c_master/u_i2c_master/p_4_in ),
        .I1(\u_i2c_master/u_i2c_master/cmd_mode_r_reg_n_0_[0] ),
        .I2(\u_i2c_master/u_i2c_master/p_1_in ),
        .I3(\u_i2c_master/u_i2c_master/p_0_in7_in ),
        .O(\FSM_onehot_state_reg[6]_i_8_n_0 ));
  LUT4 #(
    .INIT(16'hBAAA)) 
    \FSM_onehot_state_reg[7]_i_1 
       (.I0(\FSM_onehot_state_reg[7]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/mode_read_reg ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[3] ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/mode_ping_reg ),
        .O(\FSM_onehot_state_reg[7]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h4F444444)) 
    \FSM_onehot_state_reg[7]_i_2 
       (.I0(\FSM_onehot_state_reg[7]_i_3_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I2(\FSM_onehot_state_reg[9]_i_2_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg ),
        .O(\FSM_onehot_state_reg[7]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair102" *) 
  LUT3 #(
    .INIT(8'hDF)) 
    \FSM_onehot_state_reg[7]_i_3 
       (.I0(\FSM_onehot_state_reg[7]_i_4_n_0 ),
        .I1(mode_stop_reg_i_2_n_0),
        .I2(\FSM_onehot_state_reg[11]_i_3_n_0 ),
        .O(\FSM_onehot_state_reg[7]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h0000000155555555)) 
    \FSM_onehot_state_reg[7]_i_4 
       (.I0(\u_i2c_master/u_i2c_master/cmd_read0 ),
        .I1(\u_i2c_master/u_i2c_master/p_4_in ),
        .I2(\u_i2c_master/u_i2c_master/cmd_mode_r_reg_n_0_[0] ),
        .I3(\u_i2c_master/u_i2c_master/p_1_in ),
        .I4(\u_i2c_master/u_i2c_master/p_0_in7_in ),
        .I5(\FSM_onehot_state_reg[6]_i_5_n_0 ),
        .O(\FSM_onehot_state_reg[7]_i_4_n_0 ));
  LUT4 #(
    .INIT(16'hFF40)) 
    \FSM_onehot_state_reg[8]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/last_reg ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/mode_write_multiple_reg ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[10] ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/data_in_ready_next ),
        .O(\FSM_onehot_state_reg[8]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h7070707070FF7070)) 
    \FSM_onehot_state_reg[8]_i_2 
       (.I0(\u_i2c_master/u_i2c_master/data_in_ready0 ),
        .I1(\u_i2c_master/u_i2c_master/data_in_valid0 ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/mode_read_reg ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[3] ),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/mode_ping_reg ),
        .O(\u_i2c_master/u_i2c_master/u_i2c_master/data_in_ready_next ));
  LUT5 #(
    .INIT(32'hFF808080)) 
    \FSM_onehot_state_reg[9]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/data_in_ready0 ),
        .I1(\u_i2c_master/u_i2c_master/data_in_valid0 ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I3(\FSM_onehot_state_reg[9]_i_2_n_0 ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .O(\FSM_onehot_state_reg[9]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair26" *) 
  LUT4 #(
    .INIT(16'hFFFE)) 
    \FSM_onehot_state_reg[9]_i_2 
       (.I0(\u_i2c_master/bit_count_reg [3]),
        .I1(\u_i2c_master/bit_count_reg [2]),
        .I2(\u_i2c_master/bit_count_reg [1]),
        .I3(\u_i2c_master/bit_count_reg [0]),
        .O(\FSM_onehot_state_reg[9]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h000C010CFFFFFFFC)) 
    \FSM_sequential_phy_state_reg[0]_i_1 
       (.I0(\FSM_sequential_phy_state_reg[0]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I4(\FSM_sequential_phy_state_reg[0]_i_3_n_0 ),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .O(\FSM_sequential_phy_state_reg[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h1055101055555555)) 
    \FSM_sequential_phy_state_reg[0]_i_2 
       (.I0(\addr_reg[6]_i_2_n_0 ),
        .I1(\FSM_onehot_state_reg[7]_i_3_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I3(\FSM_sequential_phy_state_reg[0]_i_4_n_0 ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg ),
        .I5(\FSM_sequential_phy_state_reg[0]_i_5_n_0 ),
        .O(\FSM_sequential_phy_state_reg[0]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair53" *) 
  LUT5 #(
    .INIT(32'h00001001)) 
    \FSM_sequential_phy_state_reg[0]_i_3 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I4(\FSM_onehot_state_reg[2]_i_2_n_0 ),
        .O(\FSM_sequential_phy_state_reg[0]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair56" *) 
  LUT5 #(
    .INIT(32'hFFFEFFFF)) 
    \FSM_sequential_phy_state_reg[0]_i_4 
       (.I0(\u_i2c_master/bit_count_reg [0]),
        .I1(\u_i2c_master/bit_count_reg [1]),
        .I2(\u_i2c_master/bit_count_reg [2]),
        .I3(\u_i2c_master/bit_count_reg [3]),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\FSM_sequential_phy_state_reg[0]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'h00DD0DDD)) 
    \FSM_sequential_phy_state_reg[0]_i_5 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I1(\FSM_onehot_state_reg[6]_i_3_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .I3(\FSM_onehot_state_reg[9]_i_2_n_0 ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .O(\FSM_sequential_phy_state_reg[0]_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT5 #(
    .INIT(32'h0FFF11F0)) 
    \FSM_sequential_phy_state_reg[1]_i_1 
       (.I0(\FSM_sequential_phy_state_reg[1]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .O(\FSM_sequential_phy_state_reg[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair101" *) 
  LUT4 #(
    .INIT(16'hAA08)) 
    \FSM_sequential_phy_state_reg[1]_i_2 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_4_n_0 ),
        .I2(\FSM_sequential_phy_state_reg[0]_i_2_n_0 ),
        .I3(\FSM_sequential_phy_state_reg[0]_i_3_n_0 ),
        .O(\FSM_sequential_phy_state_reg[1]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h5A905A9058905A90)) 
    \FSM_sequential_phy_state_reg[2]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I4(\FSM_sequential_phy_state_reg[3]_i_5_n_0 ),
        .I5(\FSM_sequential_phy_state_reg[2]_i_2_n_0 ),
        .O(\FSM_sequential_phy_state_reg[2]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000010101015)) 
    \FSM_sequential_phy_state_reg[2]_i_2 
       (.I0(\FSM_onehot_state_reg[0]_i_3_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .I2(\FSM_onehot_state_reg[9]_i_2_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .I5(\addr_reg[6]_i_2_n_0 ),
        .O(\FSM_sequential_phy_state_reg[2]_i_2_n_0 ));
  LUT2 #(
    .INIT(4'h2)) 
    \FSM_sequential_phy_state_reg[3]_i_1 
       (.I0(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/delay_scl_reg ),
        .O(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_next ));
  (* SOFT_HLUTNM = "soft_lutpair55" *) 
  LUT5 #(
    .INIT(32'h55540054)) 
    \FSM_sequential_phy_state_reg[3]_i_10 
       (.I0(\addr_reg[6]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .I3(\FSM_onehot_state_reg[9]_i_2_n_0 ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\FSM_sequential_phy_state_reg[3]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'h0CC8F0F0F0F00000)) 
    \FSM_sequential_phy_state_reg[3]_i_2 
       (.I0(\FSM_sequential_phy_state_reg[3]_i_4_n_0 ),
        .I1(\FSM_sequential_phy_state_reg[3]_i_5_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .O(\FSM_sequential_phy_state_reg[3]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000002)) 
    \FSM_sequential_phy_state_reg[3]_i_3 
       (.I0(\FSM_sequential_phy_state_reg[3]_i_6_n_0 ),
        .I1(\FSM_sequential_phy_state_reg[3]_i_7_n_0 ),
        .I2(\FSM_sequential_phy_state_reg[3]_i_8_n_0 ),
        .I3(\u_i2c_master/delay_reg [11]),
        .I4(\u_i2c_master/delay_reg [1]),
        .I5(\u_i2c_master/delay_reg [15]),
        .O(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hFFFF00F4)) 
    \FSM_sequential_phy_state_reg[3]_i_4 
       (.I0(\FSM_onehot_state_reg[7]_i_3_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[11] ),
        .I2(\FSM_sequential_phy_state_reg[3]_i_9_n_0 ),
        .I3(\addr_reg[6]_i_2_n_0 ),
        .I4(\FSM_sequential_phy_state_reg[3]_i_10_n_0 ),
        .O(\FSM_sequential_phy_state_reg[3]_i_4_n_0 ));
  LUT2 #(
    .INIT(4'h1)) 
    \FSM_sequential_phy_state_reg[3]_i_5 
       (.I0(\FSM_sequential_phy_state_reg[0]_i_3_n_0 ),
        .I1(\FSM_sequential_phy_state_reg[0]_i_2_n_0 ),
        .O(\FSM_sequential_phy_state_reg[3]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000001)) 
    \FSM_sequential_phy_state_reg[3]_i_6 
       (.I0(\u_i2c_master/delay_reg [6]),
        .I1(\u_i2c_master/delay_reg [7]),
        .I2(\u_i2c_master/delay_reg [10]),
        .I3(\u_i2c_master/delay_reg [0]),
        .I4(\u_i2c_master/delay_reg [16]),
        .I5(\u_i2c_master/delay_reg [9]),
        .O(\FSM_sequential_phy_state_reg[3]_i_6_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \FSM_sequential_phy_state_reg[3]_i_7 
       (.I0(\u_i2c_master/delay_reg [2]),
        .I1(\u_i2c_master/delay_reg [5]),
        .I2(\u_i2c_master/delay_reg [3]),
        .I3(\u_i2c_master/delay_reg [14]),
        .O(\FSM_sequential_phy_state_reg[3]_i_7_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \FSM_sequential_phy_state_reg[3]_i_8 
       (.I0(\u_i2c_master/delay_reg [8]),
        .I1(\u_i2c_master/delay_reg [12]),
        .I2(\u_i2c_master/delay_reg [4]),
        .I3(\u_i2c_master/delay_reg [13]),
        .O(\FSM_sequential_phy_state_reg[3]_i_8_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair4" *) 
  LUT5 #(
    .INIT(32'hEAAAEAEA)) 
    \FSM_sequential_phy_state_reg[3]_i_9 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[7] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[10] ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/last_reg ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/mode_write_multiple_reg ),
        .O(\FSM_sequential_phy_state_reg[3]_i_9_n_0 ));
  LUT5 #(
    .INIT(32'h07FF0700)) 
    \FSM_sequential_state[0]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/state [0]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/busy_reg_reg_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/state [1]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\FSM_sequential_state[0]_i_2_n_0 ),
        .O(\FSM_sequential_state[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFBF9F9F9FFFFAEAE)) 
    \FSM_sequential_state[0]_i_2 
       (.I0(\u_i2c_master/u_i2c_master/state [1]),
        .I1(\u_i2c_master/u_i2c_master/state [3]),
        .I2(\u_i2c_master/u_i2c_master/state [2]),
        .I3(\u_i2c_master/u_i2c_master/p_1_in ),
        .I4(\u_i2c_master/u_i2c_master/data_in_last ),
        .I5(\u_i2c_master/u_i2c_master/state [0]),
        .O(\FSM_sequential_state[0]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h1111111111110001)) 
    \FSM_sequential_state[1]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/cmd_ping09_out ),
        .I1(\u_i2c_master/u_i2c_master/state [1]),
        .I2(\u_i2c_master/u_i2c_master/state [0]),
        .I3(\u_i2c_master/u_i2c_master/data_in_last ),
        .I4(\u_i2c_master/u_i2c_master/state [2]),
        .I5(\u_i2c_master/u_i2c_master/state [3]),
        .O(\FSM_sequential_state[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair25" *) 
  LUT5 #(
    .INIT(32'h00000002)) 
    \FSM_sequential_state[1]_i_2 
       (.I0(\FSM_onehot_state_reg[6]_i_5_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/p_0_in7_in ),
        .I2(\u_i2c_master/u_i2c_master/p_1_in ),
        .I3(\u_i2c_master/u_i2c_master/cmd_mode_r_reg_n_0_[0] ),
        .I4(\u_i2c_master/u_i2c_master/p_4_in ),
        .O(\u_i2c_master/u_i2c_master/cmd_ping09_out ));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT5 #(
    .INIT(32'h0000F5FD)) 
    \FSM_sequential_state[2]_i_1 
       (.I0(\FSM_sequential_state[2]_i_2_n_0 ),
        .I1(\FSM_sequential_state[2]_i_3_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/state [2]),
        .I3(\u_i2c_master/u_i2c_master/state [3]),
        .I4(\FSM_sequential_state[3]_i_6_n_0 ),
        .O(\FSM_sequential_state[2]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h404040FFFF55FFFF)) 
    \FSM_sequential_state[2]_i_2 
       (.I0(\u_i2c_master/u_i2c_master/cmd_ping09_out ),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/p_0_in7_in ),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/u_i2c_master/state [3]),
        .I5(\u_i2c_master/u_i2c_master/state [1]),
        .O(\FSM_sequential_state[2]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair103" *) 
  LUT3 #(
    .INIT(8'h2A)) 
    \FSM_sequential_state[2]_i_3 
       (.I0(\u_i2c_master/u_i2c_master/state [0]),
        .I1(\u_i2c_master/u_i2c_master/p_1_in ),
        .I2(\u_i2c_master/u_i2c_master/state [1]),
        .O(\FSM_sequential_state[2]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFAAAE)) 
    \FSM_sequential_state[3]_i_1 
       (.I0(\FSM_sequential_state[3]_i_3_n_0 ),
        .I1(\FSM_onehot_state_reg[11]_i_3_n_0 ),
        .I2(\FSM_sequential_state[3]_i_4_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/state [1]),
        .I4(\FSM_sequential_state[3]_i_5_n_0 ),
        .I5(\u_i2c_master/u_i2c_master/cmd_mode_r ),
        .O(\FSM_sequential_state[3]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hBAAAAAAABAAABAAA)) 
    \FSM_sequential_state[3]_i_2 
       (.I0(\FSM_sequential_state[3]_i_6_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [0]),
        .I4(\u_i2c_master/u_i2c_master/p_0_in7_in ),
        .I5(\u_i2c_master/u_i2c_master/state [1]),
        .O(\FSM_sequential_state[3]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT5 #(
    .INIT(32'h32FF3200)) 
    \FSM_sequential_state[3]_i_3 
       (.I0(\u_i2c_master/u_i2c_master/data_out_last ),
        .I1(\FSM_sequential_state[3]_i_7_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/state [0]),
        .I3(\u_i2c_master/u_i2c_master/state [3]),
        .I4(mem_reg_0_3_0_5_i_1__0_n_0),
        .O(\FSM_sequential_state[3]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair11" *) 
  LUT2 #(
    .INIT(4'h1)) 
    \FSM_sequential_state[3]_i_4 
       (.I0(\u_i2c_master/u_i2c_master/state [3]),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .O(\FSM_sequential_state[3]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h0015000000040000)) 
    \FSM_sequential_state[3]_i_5 
       (.I0(\u_i2c_master/u_i2c_master/state [1]),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/busy_reg_reg_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/state [3]),
        .I4(\u_i2c_master/u_i2c_master/state [2]),
        .I5(\u_i2c_master/u_i2c_master/cmd_ready0 ),
        .O(\FSM_sequential_state[3]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'hC4C4C404)) 
    \FSM_sequential_state[3]_i_6 
       (.I0(\u_i2c_master/u_i2c_master/state [0]),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/u_i2c_master/state [1]),
        .I3(\u_i2c_master/i2c_cmd_mode_reg_n_0_[2] ),
        .I4(\u_i2c_master/i2c_cmd_mode_reg_n_0_[3] ),
        .O(\FSM_sequential_state[3]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair54" *) 
  LUT4 #(
    .INIT(16'hFF7F)) 
    \FSM_sequential_state[3]_i_7 
       (.I0(\u_i2c_master/u_i2c_master/data_out_valid ),
        .I1(\u_i2c_master/u_i2c_master/state [1]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .O(\FSM_sequential_state[3]_i_7_n_0 ));
  GND GND
       (.G(\<const0> ));
  LUT6 #(
    .INIT(64'hBBBBBBBBBBB8BBBB)) 
    \__do_out[0]_i_1 
       (.I0(scratch[0]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[0]_i_2_n_0 ),
        .I3(\__do_out[0]_i_3_n_0 ),
        .I4(\__do_out[0]_i_4_n_0 ),
        .I5(\__do_out[0]_i_5_n_0 ),
        .O(p_1_in__0[0]));
  LUT6 #(
    .INIT(64'h0000000000005100)) 
    \__do_out[0]_i_10 
       (.I0(\__do_out[0]_i_16_n_0 ),
        .I1(\jtag_ctrl_reg_n_0_[0] ),
        .I2(\__do_out[4]_i_13_n_0 ),
        .I3(\__do_out[0]_i_7_n_0 ),
        .I4(\__do_out[0]_i_17_n_0 ),
        .I5(\__do_out[0]_i_18_n_0 ),
        .O(\__do_out[0]_i_10_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair106" *) 
  LUT2 #(
    .INIT(4'h7)) 
    \__do_out[0]_i_11 
       (.I0(rfmod_in[1]),
        .I1(\__do_out[15]_i_14_n_0 ),
        .O(\__do_out[0]_i_11_n_0 ));
  LUT6 #(
    .INIT(64'h45FF4555FFFFFFFF)) 
    \__do_out[0]_i_12 
       (.I0(test_en),
        .I1(rst),
        .I2(dut_pgood),
        .I3(\gpio_dir_reg_n_0_[1] ),
        .I4(test_dir),
        .I5(\__do_out[15]_i_15_n_0 ),
        .O(\__do_out[0]_i_12_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT5 #(
    .INIT(32'h00000020)) 
    \__do_out[0]_i_13 
       (.I0(ctrl2_rd[0]),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[0]_i_13_n_0 ));
  LUT6 #(
    .INIT(64'hFFCDFFFFFFFDFFFF)) 
    \__do_out[0]_i_14 
       (.I0(ctrl0_rd[0]),
        .I1(\__do_out[15]_i_28_n_0 ),
        .I2(addr[0]),
        .I3(wr),
        .I4(en),
        .I5(ctrl1_rd[0]),
        .O(\__do_out[0]_i_14_n_0 ));
  LUT6 #(
    .INIT(64'h0008000C00080000)) 
    \__do_out[0]_i_15 
       (.I0(rfmod_in[1]),
        .I1(en),
        .I2(wr),
        .I3(\__do_out[15]_i_30_n_0 ),
        .I4(addr[0]),
        .I5(rfmod_id[0]),
        .O(\__do_out[0]_i_15_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair141" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[0]_i_16 
       (.I0(i2c_data_o1[0]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[0]_i_16_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair169" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \__do_out[0]_i_17 
       (.I0(i2c_data_o0[0]),
        .I1(\__do_out[14]_i_19_n_0 ),
        .O(\__do_out[0]_i_17_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair182" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[0]_i_18 
       (.I0(\__do_out[13]_i_11_n_0 ),
        .I1(test_stat__0[16]),
        .O(\__do_out[0]_i_18_n_0 ));
  LUT6 #(
    .INIT(64'hFFFF0E000E000E00)) 
    \__do_out[0]_i_2 
       (.I0(gpio_exp_busy),
        .I1(addr[0]),
        .I2(\__do_out[0]_i_6_n_0 ),
        .I3(gpio_exp_trigger_i_2_n_0),
        .I4(gpio_exp_rb[16]),
        .I5(\__do_out[7]_i_2_n_0 ),
        .O(\__do_out[0]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair164" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \__do_out[0]_i_3 
       (.I0(\^cbrs_rev ),
        .I1(\__do_out[14]_i_6_n_0 ),
        .O(\__do_out[0]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair50" *) 
  LUT5 #(
    .INIT(32'h00005DDD)) 
    \__do_out[0]_i_4 
       (.I0(rst),
        .I1(\__do_out[0]_i_7_n_0 ),
        .I2(\__do_out[15]_i_12_n_0 ),
        .I3(test_stat__0[0]),
        .I4(\__do_out[0]_i_8_n_0 ),
        .O(\__do_out[0]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h8A88AAAAAAAAAAAA)) 
    \__do_out[0]_i_5 
       (.I0(\__do_out[14]_i_5_n_0 ),
        .I1(\__do_out[0]_i_9_n_0 ),
        .I2(\__do_out[0]_i_10_n_0 ),
        .I3(\__do_out[14]_i_9_n_0 ),
        .I4(\__do_out[0]_i_11_n_0 ),
        .I5(\__do_out[0]_i_12_n_0 ),
        .O(\__do_out[0]_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair39" *) 
  LUT2 #(
    .INIT(4'hB)) 
    \__do_out[0]_i_6 
       (.I0(wr),
        .I1(en),
        .O(\__do_out[0]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'h0000000075557575)) 
    \__do_out[0]_i_7 
       (.I0(\__do_out[15]_i_20_n_0 ),
        .I1(\__do_out[0]_i_13_n_0 ),
        .I2(\__do_out[0]_i_14_n_0 ),
        .I3(\__do_out[15]_i_19_n_0 ),
        .I4(\ctrl3_reg_n_0_[0] ),
        .I5(\__do_out[0]_i_15_n_0 ),
        .O(\__do_out[0]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'hA0ACACA000000000)) 
    \__do_out[0]_i_8 
       (.I0(\gpio_dato_reg_n_0_[1] ),
        .I1(\rfmod_out[9]_INST_0_i_3_n_0 ),
        .I2(\gpio_dir_reg_n_0_[1] ),
        .I3(\u_tester/u_gen/out_reg_n_0_[0] ),
        .I4(test_stat[23]),
        .I5(\__do_out[15]_i_10_n_0 ),
        .O(\__do_out[0]_i_8_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair50" *) 
  LUT4 #(
    .INIT(16'h8F88)) 
    \__do_out[0]_i_9 
       (.I0(test_stat__0[0]),
        .I1(\__do_out[15]_i_12_n_0 ),
        .I2(\__do_out[0]_i_7_n_0 ),
        .I3(rst),
        .O(\__do_out[0]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hB8BBBBBBB8BBB8BB)) 
    \__do_out[10]_i_1 
       (.I0(scratch[10]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[10]_i_2_n_0 ),
        .I3(\__do_out[10]_i_3_n_0 ),
        .I4(\__do_out[10]_i_4_n_0 ),
        .I5(\__do_out[14]_i_5_n_0 ),
        .O(p_1_in__0[10]));
  LUT6 #(
    .INIT(64'hD0DD0000D0DDD0DD)) 
    \__do_out[10]_i_10 
       (.I0(ctrl0_rd[10]),
        .I1(\__do_out[13]_i_13_n_0 ),
        .I2(\__do_out[13]_i_14_n_0 ),
        .I3(ctrl1_rd[10]),
        .I4(\__do_out[13]_i_15_n_0 ),
        .I5(ctrl2_rd[10]),
        .O(\__do_out[10]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'h4F4F4F444F4F4F4F)) 
    \__do_out[10]_i_11 
       (.I0(\__do_out[13]_i_16_n_0 ),
        .I1(rfmod_in[11]),
        .I2(\__do_out[13]_i_17_n_0 ),
        .I3(p_0_in0_in),
        .I4(tx_active[0]),
        .I5(wr_cache_reg_0_7_6_7_i_3_n_0),
        .O(\__do_out[10]_i_11_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \__do_out[10]_i_2 
       (.I0(\__do_out[7]_i_2_n_0 ),
        .I1(gpio_exp_rb[26]),
        .I2(\__do_out[15]_i_8_n_0 ),
        .I3(p_29_in[10]),
        .O(\__do_out[10]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000007007777)) 
    \__do_out[10]_i_3 
       (.I0(rfmod_out[11]),
        .I1(\__do_out[15]_i_10_n_0 ),
        .I2(\__do_out[10]_i_5_n_0 ),
        .I3(\__do_out[10]_i_6_n_0 ),
        .I4(rst),
        .I5(\__do_out[10]_i_7_n_0 ),
        .O(\__do_out[10]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h000000000DDD0000)) 
    \__do_out[10]_i_4 
       (.I0(\__do_out[15]_i_15_n_0 ),
        .I1(\__do_out[10]_i_8_n_0 ),
        .I2(rfmod_in[11]),
        .I3(\__do_out[15]_i_14_n_0 ),
        .I4(\__do_out[10]_i_6_n_0 ),
        .I5(\__do_out[10]_i_5_n_0 ),
        .O(\__do_out[10]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFF8F8F888888888)) 
    \__do_out[10]_i_5 
       (.I0(\__do_out[15]_i_12_n_0 ),
        .I1(test_stat__0[10]),
        .I2(\__do_out[13]_i_11_n_0 ),
        .I3(\__do_out[14]_i_17_n_0 ),
        .I4(i2c_data_o0[2]),
        .I5(\__do_out[10]_i_9_n_0 ),
        .O(\__do_out[10]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h00008AFF)) 
    \__do_out[10]_i_6 
       (.I0(\__do_out[10]_i_10_n_0 ),
        .I1(\__do_out[15]_i_19_n_0 ),
        .I2(\ctrl3_reg_n_0_[10] ),
        .I3(\__do_out[15]_i_20_n_0 ),
        .I4(\__do_out[10]_i_11_n_0 ),
        .O(\__do_out[10]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair168" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[10]_i_7 
       (.I0(gpio_exp_rb[10]),
        .I1(\__do_out[15]_i_9_n_0 ),
        .O(\__do_out[10]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'h0F0FBF000000BF00)) 
    \__do_out[10]_i_8 
       (.I0(rst),
        .I1(dut_pgood),
        .I2(\gpio_dir_reg_n_0_[11] ),
        .I3(scl_t),
        .I4(test_en),
        .I5(test_dir),
        .O(\__do_out[10]_i_8_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT5 #(
    .INIT(32'h01110101)) 
    \__do_out[10]_i_9 
       (.I0(\__do_out[15]_i_12_n_0 ),
        .I1(rst),
        .I2(\__do_out[13]_i_11_n_0 ),
        .I3(\u_tester/st_cur [1]),
        .I4(\u_tester/st_cur [0]),
        .O(\__do_out[10]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hB8BBBBBBB8BBB8BB)) 
    \__do_out[11]_i_1 
       (.I0(scratch[11]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[11]_i_2_n_0 ),
        .I3(\__do_out[11]_i_3_n_0 ),
        .I4(\__do_out[11]_i_4_n_0 ),
        .I5(\__do_out[14]_i_5_n_0 ),
        .O(p_1_in__0[11]));
  LUT6 #(
    .INIT(64'h4444444F44444444)) 
    \__do_out[11]_i_10 
       (.I0(\__do_out[13]_i_16_n_0 ),
        .I1(rfmod_in[12]),
        .I2(\__do_out[13]_i_17_n_0 ),
        .I3(p_0_in0_in),
        .I4(tx_active[0]),
        .I5(wr_cache_reg_0_7_6_7_i_3_n_0),
        .O(\__do_out[11]_i_10_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \__do_out[11]_i_2 
       (.I0(\__do_out[7]_i_2_n_0 ),
        .I1(gpio_exp_rb[27]),
        .I2(\__do_out[15]_i_8_n_0 ),
        .I3(p_29_in[11]),
        .O(\__do_out[11]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000007007777)) 
    \__do_out[11]_i_3 
       (.I0(rfmod_out[12]),
        .I1(\__do_out[15]_i_10_n_0 ),
        .I2(\__do_out[11]_i_5_n_0 ),
        .I3(\__do_out[11]_i_6_n_0 ),
        .I4(rst),
        .I5(\__do_out[11]_i_7_n_0 ),
        .O(\__do_out[11]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h0000000007770000)) 
    \__do_out[11]_i_4 
       (.I0(\__do_out[15]_i_15_n_0 ),
        .I1(rfmod_oe[12]),
        .I2(rfmod_in[12]),
        .I3(\__do_out[15]_i_14_n_0 ),
        .I4(\__do_out[11]_i_6_n_0 ),
        .I5(\__do_out[11]_i_5_n_0 ),
        .O(\__do_out[11]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAA00AAFCAA00AA30)) 
    \__do_out[11]_i_5 
       (.I0(test_stat__0[11]),
        .I1(\__do_out[13]_i_11_n_0 ),
        .I2(\__do_out[11]_i_8_n_0 ),
        .I3(\__do_out[15]_i_12_n_0 ),
        .I4(rst),
        .I5(test_stat[27]),
        .O(\__do_out[11]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h0000D5DD)) 
    \__do_out[11]_i_6 
       (.I0(\__do_out[15]_i_20_n_0 ),
        .I1(\__do_out[11]_i_9_n_0 ),
        .I2(\__do_out[15]_i_19_n_0 ),
        .I3(\ctrl3_reg_n_0_[11] ),
        .I4(\__do_out[11]_i_10_n_0 ),
        .O(\__do_out[11]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair161" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[11]_i_7 
       (.I0(gpio_exp_rb[11]),
        .I1(\__do_out[15]_i_9_n_0 ),
        .O(\__do_out[11]_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair181" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[11]_i_8 
       (.I0(i2c_data_o0[3]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[11]_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hD0DD0000D0DDD0DD)) 
    \__do_out[11]_i_9 
       (.I0(ctrl0_rd[11]),
        .I1(\__do_out[13]_i_13_n_0 ),
        .I2(\__do_out[13]_i_14_n_0 ),
        .I3(ctrl1_rd[11]),
        .I4(\__do_out[13]_i_15_n_0 ),
        .I5(ctrl2_rd[11]),
        .O(\__do_out[11]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hB8BBBBBBB8BBB8BB)) 
    \__do_out[12]_i_1 
       (.I0(scratch[12]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[12]_i_2_n_0 ),
        .I3(\__do_out[12]_i_3_n_0 ),
        .I4(\__do_out[12]_i_4_n_0 ),
        .I5(\__do_out[14]_i_5_n_0 ),
        .O(p_1_in__0[12]));
  LUT4 #(
    .INIT(16'h4F44)) 
    \__do_out[12]_i_10 
       (.I0(\__do_out[13]_i_16_n_0 ),
        .I1(rfmod_in[13]),
        .I2(\__do_out[13]_i_17_n_0 ),
        .I3(trx_auto),
        .O(\__do_out[12]_i_10_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \__do_out[12]_i_2 
       (.I0(\__do_out[7]_i_2_n_0 ),
        .I1(gpio_exp_rb[28]),
        .I2(\__do_out[15]_i_8_n_0 ),
        .I3(p_29_in[12]),
        .O(\__do_out[12]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h000000000DDD0D0D)) 
    \__do_out[12]_i_3 
       (.I0(gpio_exp_rb[12]),
        .I1(\__do_out[14]_i_6_n_0 ),
        .I2(rst),
        .I3(\__do_out[12]_i_5_n_0 ),
        .I4(\__do_out[12]_i_6_n_0 ),
        .I5(\__do_out[12]_i_7_n_0 ),
        .O(\__do_out[12]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h0000000007770000)) 
    \__do_out[12]_i_4 
       (.I0(\__do_out[15]_i_15_n_0 ),
        .I1(rfmod_oe[13]),
        .I2(rfmod_in[13]),
        .I3(\__do_out[15]_i_14_n_0 ),
        .I4(\__do_out[12]_i_6_n_0 ),
        .I5(\__do_out[12]_i_5_n_0 ),
        .O(\__do_out[12]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAA00AAFCAA00AA30)) 
    \__do_out[12]_i_5 
       (.I0(test_stat__0[12]),
        .I1(\__do_out[13]_i_11_n_0 ),
        .I2(\__do_out[12]_i_8_n_0 ),
        .I3(\__do_out[15]_i_12_n_0 ),
        .I4(rst),
        .I5(test_en),
        .O(\__do_out[12]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h0000D5DD)) 
    \__do_out[12]_i_6 
       (.I0(\__do_out[15]_i_20_n_0 ),
        .I1(\__do_out[12]_i_9_n_0 ),
        .I2(\__do_out[15]_i_19_n_0 ),
        .I3(\ctrl3_reg_n_0_[12] ),
        .I4(\__do_out[12]_i_10_n_0 ),
        .O(\__do_out[12]_i_6_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[12]_i_7 
       (.I0(rfmod_out[13]),
        .I1(\__do_out[15]_i_10_n_0 ),
        .O(\__do_out[12]_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair186" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[12]_i_8 
       (.I0(i2c_data_o0[4]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[12]_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hD0DD0000D0DDD0DD)) 
    \__do_out[12]_i_9 
       (.I0(ctrl0_rd[12]),
        .I1(\__do_out[13]_i_13_n_0 ),
        .I2(\__do_out[13]_i_14_n_0 ),
        .I3(ctrl1_rd[12]),
        .I4(\__do_out[13]_i_15_n_0 ),
        .I5(ctrl2_rd[12]),
        .O(\__do_out[12]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hBBBBBBBBBBB8BBBB)) 
    \__do_out[13]_i_1 
       (.I0(scratch[13]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[13]_i_2_n_0 ),
        .I3(\__do_out[13]_i_3_n_0 ),
        .I4(\__do_out[13]_i_4_n_0 ),
        .I5(\__do_out[13]_i_5_n_0 ),
        .O(p_1_in__0[13]));
  LUT4 #(
    .INIT(16'h4F44)) 
    \__do_out[13]_i_10 
       (.I0(\__do_out[13]_i_16_n_0 ),
        .I1(rfmod_in[14]),
        .I2(\__do_out[13]_i_17_n_0 ),
        .I3(id_changed),
        .O(\__do_out[13]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000100000)) 
    \__do_out[13]_i_11 
       (.I0(addr[2]),
        .I1(addr[5]),
        .I2(addr[3]),
        .I3(addr[4]),
        .I4(addr[1]),
        .I5(\__do_out[7]_i_5_n_0 ),
        .O(\__do_out[13]_i_11_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair57" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[13]_i_12 
       (.I0(i2c_data_o0[5]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[13]_i_12_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    \__do_out[13]_i_13 
       (.I0(addr[4]),
        .I1(addr[3]),
        .I2(addr[5]),
        .I3(addr[2]),
        .I4(addr[1]),
        .I5(\__do_out[15]_i_16_n_0 ),
        .O(\__do_out[13]_i_13_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    \__do_out[13]_i_14 
       (.I0(addr[4]),
        .I1(addr[3]),
        .I2(addr[5]),
        .I3(addr[2]),
        .I4(addr[1]),
        .I5(\__do_out[7]_i_5_n_0 ),
        .O(\__do_out[13]_i_14_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFD)) 
    \__do_out[13]_i_15 
       (.I0(addr[1]),
        .I1(addr[4]),
        .I2(addr[3]),
        .I3(addr[5]),
        .I4(addr[2]),
        .I5(\__do_out[15]_i_16_n_0 ),
        .O(\__do_out[13]_i_15_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT5 #(
    .INIT(32'hFFFFFFDF)) 
    \__do_out[13]_i_16 
       (.I0(en),
        .I1(wr),
        .I2(addr[0]),
        .I3(addr[1]),
        .I4(\__do_out[15]_i_29_n_0 ),
        .O(\__do_out[13]_i_16_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair33" *) 
  LUT5 #(
    .INIT(32'hFFFFFFFD)) 
    \__do_out[13]_i_17 
       (.I0(en),
        .I1(wr),
        .I2(addr[1]),
        .I3(\__do_out[15]_i_29_n_0 ),
        .I4(addr[0]),
        .O(\__do_out[13]_i_17_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \__do_out[13]_i_2 
       (.I0(\__do_out[7]_i_2_n_0 ),
        .I1(gpio_exp_rb[29]),
        .I2(\__do_out[15]_i_8_n_0 ),
        .I3(p_29_in[13]),
        .O(\__do_out[13]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair149" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[13]_i_3 
       (.I0(gpio_exp_rb[13]),
        .I1(\__do_out[15]_i_9_n_0 ),
        .O(\__do_out[13]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'h005D5D5D)) 
    \__do_out[13]_i_4 
       (.I0(rst),
        .I1(\__do_out[13]_i_6_n_0 ),
        .I2(\__do_out[13]_i_7_n_0 ),
        .I3(\__do_out[15]_i_10_n_0 ),
        .I4(rfmod_out[14]),
        .O(\__do_out[13]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAA8A8A8AAAAAAAAA)) 
    \__do_out[13]_i_5 
       (.I0(\__do_out[14]_i_5_n_0 ),
        .I1(\__do_out[13]_i_7_n_0 ),
        .I2(\__do_out[13]_i_6_n_0 ),
        .I3(\__do_out[15]_i_14_n_0 ),
        .I4(rfmod_in[14]),
        .I5(\__do_out[13]_i_8_n_0 ),
        .O(\__do_out[13]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h00008AFF)) 
    \__do_out[13]_i_6 
       (.I0(\__do_out[13]_i_9_n_0 ),
        .I1(\__do_out[15]_i_19_n_0 ),
        .I2(\ctrl3_reg_n_0_[13] ),
        .I3(\__do_out[15]_i_20_n_0 ),
        .I4(\__do_out[13]_i_10_n_0 ),
        .O(\__do_out[13]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'hAA00AAFCAA00AA30)) 
    \__do_out[13]_i_7 
       (.I0(test_stat__0[13]),
        .I1(\__do_out[13]_i_11_n_0 ),
        .I2(\__do_out[13]_i_12_n_0 ),
        .I3(\__do_out[15]_i_12_n_0 ),
        .I4(rst),
        .I5(test_stat[29]),
        .O(\__do_out[13]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'hFFFF1504FFFFFFFF)) 
    \__do_out[13]_i_8 
       (.I0(\gpio_dir_reg_n_0_[14] ),
        .I1(test_en),
        .I2(test_dir),
        .I3(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I4(\rfmod_oe[13]_INST_0_i_1_n_0 ),
        .I5(\__do_out[15]_i_15_n_0 ),
        .O(\__do_out[13]_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hD0DD0000D0DDD0DD)) 
    \__do_out[13]_i_9 
       (.I0(ctrl0_rd[13]),
        .I1(\__do_out[13]_i_13_n_0 ),
        .I2(\__do_out[13]_i_14_n_0 ),
        .I3(ctrl1_rd[13]),
        .I4(\__do_out[13]_i_15_n_0 ),
        .I5(ctrl2_rd[13]),
        .O(\__do_out[13]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hB8BBBBBBB8BBB8BB)) 
    \__do_out[14]_i_1 
       (.I0(scratch[14]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[14]_i_2_n_0 ),
        .I3(\__do_out[14]_i_3_n_0 ),
        .I4(\__do_out[14]_i_4_n_0 ),
        .I5(\__do_out[14]_i_5_n_0 ),
        .O(p_1_in__0[14]));
  LUT6 #(
    .INIT(64'h2A002A0000002A00)) 
    \__do_out[14]_i_10 
       (.I0(\__do_out[15]_i_24_n_0 ),
        .I1(i2c_data_o0[6]),
        .I2(\__do_out[14]_i_17_n_0 ),
        .I3(\__do_out[14]_i_18_n_0 ),
        .I4(i2c_error),
        .I5(\__do_out[14]_i_19_n_0 ),
        .O(\__do_out[14]_i_10_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair72" *) 
  LUT4 #(
    .INIT(16'h8F88)) 
    \__do_out[14]_i_11 
       (.I0(\__do_out[15]_i_12_n_0 ),
        .I1(test_stat__0[14]),
        .I2(\__do_out[14]_i_18_n_0 ),
        .I3(rst),
        .O(\__do_out[14]_i_11_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT5 #(
    .INIT(32'hFFFFFFFB)) 
    \__do_out[14]_i_12 
       (.I0(addr[1]),
        .I1(addr[4]),
        .I2(addr[2]),
        .I3(addr[5]),
        .I4(addr[3]),
        .O(\__do_out[14]_i_12_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair186" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[14]_i_13 
       (.I0(i2c_data_o0[6]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[14]_i_13_n_0 ));
  LUT6 #(
    .INIT(64'hFFEFAAAAFFFFFFFF)) 
    \__do_out[14]_i_14 
       (.I0(\__do_out[14]_i_20_n_0 ),
        .I1(\__do_out[14]_i_21_n_0 ),
        .I2(\__do_out[14]_i_22_n_0 ),
        .I3(\__do_out[14]_i_23_n_0 ),
        .I4(\__do_out[15]_i_20_n_0 ),
        .I5(\__do_out[14]_i_24_n_0 ),
        .O(\__do_out[14]_i_14_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAA8AA88888888)) 
    \__do_out[14]_i_15 
       (.I0(rst),
        .I1(\__do_out[14]_i_20_n_0 ),
        .I2(\__do_out[14]_i_21_n_0 ),
        .I3(\__do_out[14]_i_22_n_0 ),
        .I4(\__do_out[14]_i_23_n_0 ),
        .I5(\__do_out[15]_i_20_n_0 ),
        .O(\__do_out[14]_i_15_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair72" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[14]_i_16 
       (.I0(test_stat__0[14]),
        .I1(\__do_out[15]_i_12_n_0 ),
        .O(\__do_out[14]_i_16_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000100000)) 
    \__do_out[14]_i_17 
       (.I0(addr[2]),
        .I1(addr[5]),
        .I2(addr[3]),
        .I3(addr[4]),
        .I4(addr[1]),
        .I5(\__do_out[15]_i_16_n_0 ),
        .O(\__do_out[14]_i_17_n_0 ));
  LUT6 #(
    .INIT(64'h0000000075557575)) 
    \__do_out[14]_i_18 
       (.I0(\__do_out[15]_i_20_n_0 ),
        .I1(\__do_out[14]_i_23_n_0 ),
        .I2(\__do_out[14]_i_22_n_0 ),
        .I3(\__do_out[15]_i_19_n_0 ),
        .I4(\ctrl3_reg_n_0_[14] ),
        .I5(\__do_out[14]_i_20_n_0 ),
        .O(\__do_out[14]_i_18_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFEF)) 
    \__do_out[14]_i_19 
       (.I0(addr[2]),
        .I1(addr[5]),
        .I2(addr[3]),
        .I3(addr[4]),
        .I4(addr[1]),
        .I5(\__do_out[15]_i_16_n_0 ),
        .O(\__do_out[14]_i_19_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \__do_out[14]_i_2 
       (.I0(\__do_out[7]_i_2_n_0 ),
        .I1(gpio_exp_rb[30]),
        .I2(\__do_out[15]_i_8_n_0 ),
        .I3(p_29_in[14]),
        .O(\__do_out[14]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0008000C00080000)) 
    \__do_out[14]_i_20 
       (.I0(rfmod_in[15]),
        .I1(en),
        .I2(wr),
        .I3(\__do_out[15]_i_30_n_0 ),
        .I4(addr[0]),
        .I5(dut_pgood),
        .O(\__do_out[14]_i_20_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT5 #(
    .INIT(32'h00000800)) 
    \__do_out[14]_i_21 
       (.I0(\ctrl3_reg_n_0_[14] ),
        .I1(en),
        .I2(wr),
        .I3(addr[0]),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[14]_i_21_n_0 ));
  LUT6 #(
    .INIT(64'hFDFFFCFFFDFFFFFF)) 
    \__do_out[14]_i_22 
       (.I0(ctrl1_rd[14]),
        .I1(\__do_out[15]_i_28_n_0 ),
        .I2(wr),
        .I3(en),
        .I4(addr[0]),
        .I5(ctrl0_rd[14]),
        .O(\__do_out[14]_i_22_n_0 ));
  LUT5 #(
    .INIT(32'h00000020)) 
    \__do_out[14]_i_23 
       (.I0(ctrl2_rd[14]),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[14]_i_23_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFBFFFFFFFF)) 
    \__do_out[14]_i_24 
       (.I0(addr[0]),
        .I1(en),
        .I2(wr),
        .I3(addr[1]),
        .I4(\__do_out[15]_i_35_n_0 ),
        .I5(i2c_error),
        .O(\__do_out[14]_i_24_n_0 ));
  LUT6 #(
    .INIT(64'h0DDD0DDD00000DDD)) 
    \__do_out[14]_i_3 
       (.I0(gpio_exp_rb[14]),
        .I1(\__do_out[14]_i_6_n_0 ),
        .I2(rfmod_out[15]),
        .I3(\__do_out[15]_i_10_n_0 ),
        .I4(rst),
        .I5(\__do_out[14]_i_7_n_0 ),
        .O(\__do_out[14]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h000000A200A200A2)) 
    \__do_out[14]_i_4 
       (.I0(\__do_out[14]_i_8_n_0 ),
        .I1(\__do_out[14]_i_9_n_0 ),
        .I2(\__do_out[14]_i_10_n_0 ),
        .I3(\__do_out[14]_i_11_n_0 ),
        .I4(rfmod_in[15]),
        .I5(\__do_out[15]_i_14_n_0 ),
        .O(\__do_out[14]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair87" *) 
  LUT3 #(
    .INIT(8'h01)) 
    \__do_out[14]_i_5 
       (.I0(\__do_out[15]_i_10_n_0 ),
        .I1(\__do_out[15]_i_9_n_0 ),
        .I2(rst),
        .O(\__do_out[14]_i_5_n_0 ));
  LUT4 #(
    .INIT(16'hFFFD)) 
    \__do_out[14]_i_6 
       (.I0(en),
        .I1(wr),
        .I2(\__do_out[14]_i_12_n_0 ),
        .I3(addr[0]),
        .O(\__do_out[14]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'h000000000000555D)) 
    \__do_out[14]_i_7 
       (.I0(\__do_out[14]_i_9_n_0 ),
        .I1(\__do_out[15]_i_24_n_0 ),
        .I2(\__do_out[14]_i_13_n_0 ),
        .I3(\__do_out[14]_i_14_n_0 ),
        .I4(\__do_out[14]_i_15_n_0 ),
        .I5(\__do_out[14]_i_16_n_0 ),
        .O(\__do_out[14]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'hFFFF1504FFFFFFFF)) 
    \__do_out[14]_i_8 
       (.I0(\gpio_dir_reg_n_0_[15] ),
        .I1(test_en),
        .I2(test_dir),
        .I3(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I4(\rfmod_oe[13]_INST_0_i_1_n_0 ),
        .I5(\__do_out[15]_i_15_n_0 ),
        .O(\__do_out[14]_i_8_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair62" *) 
  LUT2 #(
    .INIT(4'h1)) 
    \__do_out[14]_i_9 
       (.I0(rst),
        .I1(\__do_out[15]_i_12_n_0 ),
        .O(\__do_out[14]_i_9_n_0 ));
  LUT2 #(
    .INIT(4'h2)) 
    \__do_out[15]_i_1 
       (.I0(en),
        .I1(wr),
        .O(\__do_out[15]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000800)) 
    \__do_out[15]_i_10 
       (.I0(addr[1]),
        .I1(addr[3]),
        .I2(addr[5]),
        .I3(addr[2]),
        .I4(addr[4]),
        .I5(\__do_out[7]_i_5_n_0 ),
        .O(\__do_out[15]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFBFBB0000)) 
    \__do_out[15]_i_11 
       (.I0(\__do_out[15]_i_17_n_0 ),
        .I1(\__do_out[15]_i_18_n_0 ),
        .I2(\__do_out[15]_i_19_n_0 ),
        .I3(\ctrl3_reg_n_0_[15] ),
        .I4(\__do_out[15]_i_20_n_0 ),
        .I5(\__do_out[15]_i_21_n_0 ),
        .O(\__do_out[15]_i_11_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000020)) 
    \__do_out[15]_i_12 
       (.I0(addr[3]),
        .I1(addr[5]),
        .I2(addr[2]),
        .I3(addr[4]),
        .I4(addr[1]),
        .I5(\__do_out[15]_i_16_n_0 ),
        .O(\__do_out[15]_i_12_n_0 ));
  LUT6 #(
    .INIT(64'hFEFEFEFEFEFEEEFE)) 
    \__do_out[15]_i_13 
       (.I0(\__do_out[15]_i_22_n_0 ),
        .I1(\__do_out[15]_i_23_n_0 ),
        .I2(\__do_out[14]_i_9_n_0 ),
        .I3(\__do_out[15]_i_24_n_0 ),
        .I4(\__do_out[15]_i_25_n_0 ),
        .I5(\__do_out[15]_i_26_n_0 ),
        .O(\__do_out[15]_i_13_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000020)) 
    \__do_out[15]_i_14 
       (.I0(addr[3]),
        .I1(addr[5]),
        .I2(addr[2]),
        .I3(addr[4]),
        .I4(addr[1]),
        .I5(\__do_out[7]_i_5_n_0 ),
        .O(\__do_out[15]_i_14_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000800)) 
    \__do_out[15]_i_15 
       (.I0(addr[1]),
        .I1(addr[3]),
        .I2(addr[5]),
        .I3(addr[2]),
        .I4(addr[4]),
        .I5(\__do_out[15]_i_16_n_0 ),
        .O(\__do_out[15]_i_15_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT3 #(
    .INIT(8'hFB)) 
    \__do_out[15]_i_16 
       (.I0(addr[0]),
        .I1(en),
        .I2(wr),
        .O(\__do_out[15]_i_16_n_0 ));
  LUT5 #(
    .INIT(32'h00000020)) 
    \__do_out[15]_i_17 
       (.I0(ctrl2_rd[15]),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[15]_i_17_n_0 ));
  LUT6 #(
    .INIT(64'hFDFFFCFFFDFFFFFF)) 
    \__do_out[15]_i_18 
       (.I0(ctrl1_rd[15]),
        .I1(\__do_out[15]_i_28_n_0 ),
        .I2(wr),
        .I3(en),
        .I4(addr[0]),
        .I5(ctrl0_rd[15]),
        .O(\__do_out[15]_i_18_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFD)) 
    \__do_out[15]_i_19 
       (.I0(addr[1]),
        .I1(addr[4]),
        .I2(addr[3]),
        .I3(addr[5]),
        .I4(addr[2]),
        .I5(\__do_out[7]_i_5_n_0 ),
        .O(\__do_out[15]_i_19_n_0 ));
  LUT6 #(
    .INIT(64'hBBBBBBBBBBB8BBBB)) 
    \__do_out[15]_i_2 
       (.I0(scratch[15]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[15]_i_4_n_0 ),
        .I3(\__do_out[15]_i_5_n_0 ),
        .I4(\__do_out[15]_i_6_n_0 ),
        .I5(\__do_out[15]_i_7_n_0 ),
        .O(p_1_in__0[15]));
  LUT5 #(
    .INIT(32'h55545555)) 
    \__do_out[15]_i_20 
       (.I0(rst),
        .I1(addr[1]),
        .I2(\__do_out[15]_i_29_n_0 ),
        .I3(wr),
        .I4(en),
        .O(\__do_out[15]_i_20_n_0 ));
  LUT6 #(
    .INIT(64'h0203020002030203)) 
    \__do_out[15]_i_21 
       (.I0(rfmod_in[16]),
        .I1(\__do_out[0]_i_6_n_0 ),
        .I2(\__do_out[15]_i_30_n_0 ),
        .I3(addr[0]),
        .I4(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I5(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .O(\__do_out[15]_i_21_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair177" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[15]_i_22 
       (.I0(test_stat__0[15]),
        .I1(\__do_out[15]_i_12_n_0 ),
        .O(\__do_out[15]_i_22_n_0 ));
  LUT6 #(
    .INIT(64'hEEEEEAEE00000000)) 
    \__do_out[15]_i_23 
       (.I0(\__do_out[15]_i_21_n_0 ),
        .I1(\__do_out[15]_i_20_n_0 ),
        .I2(\__do_out[15]_i_31_n_0 ),
        .I3(\__do_out[15]_i_18_n_0 ),
        .I4(\__do_out[15]_i_17_n_0 ),
        .I5(\__do_out[15]_i_32_n_0 ),
        .O(\__do_out[15]_i_23_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair180" *) 
  LUT2 #(
    .INIT(4'h7)) 
    \__do_out[15]_i_24 
       (.I0(test_dir),
        .I1(\__do_out[13]_i_11_n_0 ),
        .O(\__do_out[15]_i_24_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[15]_i_25 
       (.I0(i2c_data_o0[7]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[15]_i_25_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFEEEEEAEE)) 
    \__do_out[15]_i_26 
       (.I0(\__do_out[15]_i_21_n_0 ),
        .I1(\__do_out[15]_i_20_n_0 ),
        .I2(\__do_out[15]_i_31_n_0 ),
        .I3(\__do_out[15]_i_18_n_0 ),
        .I4(\__do_out[15]_i_17_n_0 ),
        .I5(\__do_out[15]_i_33_n_0 ),
        .O(\__do_out[15]_i_26_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT5 #(
    .INIT(32'hFFFEFFFF)) 
    \__do_out[15]_i_27 
       (.I0(addr[2]),
        .I1(addr[5]),
        .I2(addr[3]),
        .I3(addr[4]),
        .I4(addr[1]),
        .O(\__do_out[15]_i_27_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair40" *) 
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \__do_out[15]_i_28 
       (.I0(addr[1]),
        .I1(addr[2]),
        .I2(addr[5]),
        .I3(addr[3]),
        .I4(addr[4]),
        .O(\__do_out[15]_i_28_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair85" *) 
  LUT4 #(
    .INIT(16'hFFEF)) 
    \__do_out[15]_i_29 
       (.I0(addr[5]),
        .I1(addr[3]),
        .I2(addr[2]),
        .I3(addr[4]),
        .O(\__do_out[15]_i_29_n_0 ));
  LUT4 #(
    .INIT(16'h0010)) 
    \__do_out[15]_i_3 
       (.I0(\scratch[15]_i_2_n_0 ),
        .I1(wr),
        .I2(en),
        .I3(rst),
        .O(\__do_out[15]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT5 #(
    .INIT(32'hFFFFFFEF)) 
    \__do_out[15]_i_30 
       (.I0(addr[1]),
        .I1(addr[4]),
        .I2(addr[2]),
        .I3(addr[3]),
        .I4(addr[5]),
        .O(\__do_out[15]_i_30_n_0 ));
  LUT5 #(
    .INIT(32'h00000800)) 
    \__do_out[15]_i_31 
       (.I0(\ctrl3_reg_n_0_[15] ),
        .I1(en),
        .I2(wr),
        .I3(addr[0]),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[15]_i_31_n_0 ));
  LUT5 #(
    .INIT(32'hAAAAAA8A)) 
    \__do_out[15]_i_32 
       (.I0(rst),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(\__do_out[15]_i_34_n_0 ),
        .O(\__do_out[15]_i_32_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000020)) 
    \__do_out[15]_i_33 
       (.I0(i2c_busy),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(addr[1]),
        .I5(\__do_out[15]_i_35_n_0 ),
        .O(\__do_out[15]_i_33_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair34" *) 
  LUT5 #(
    .INIT(32'hFFEFFFFF)) 
    \__do_out[15]_i_34 
       (.I0(addr[1]),
        .I1(addr[4]),
        .I2(addr[2]),
        .I3(addr[5]),
        .I4(addr[3]),
        .O(\__do_out[15]_i_34_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair85" *) 
  LUT4 #(
    .INIT(16'hFFEF)) 
    \__do_out[15]_i_35 
       (.I0(addr[2]),
        .I1(addr[5]),
        .I2(addr[3]),
        .I3(addr[4]),
        .O(\__do_out[15]_i_35_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \__do_out[15]_i_4 
       (.I0(\__do_out[7]_i_2_n_0 ),
        .I1(gpio_exp_rb[31]),
        .I2(\__do_out[15]_i_8_n_0 ),
        .I3(p_29_in[15]),
        .O(\__do_out[15]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair168" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[15]_i_5 
       (.I0(gpio_exp_rb[15]),
        .I1(\__do_out[15]_i_9_n_0 ),
        .O(\__do_out[15]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h0707777700777777)) 
    \__do_out[15]_i_6 
       (.I0(rfmod_out[16]),
        .I1(\__do_out[15]_i_10_n_0 ),
        .I2(test_stat__0[15]),
        .I3(\__do_out[15]_i_11_n_0 ),
        .I4(rst),
        .I5(\__do_out[15]_i_12_n_0 ),
        .O(\__do_out[15]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAA888A888A888)) 
    \__do_out[15]_i_7 
       (.I0(\__do_out[14]_i_5_n_0 ),
        .I1(\__do_out[15]_i_13_n_0 ),
        .I2(\__do_out[15]_i_14_n_0 ),
        .I3(rfmod_in[16]),
        .I4(rfmod_oe[16]),
        .I5(\__do_out[15]_i_15_n_0 ),
        .O(\__do_out[15]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'h0000000001000000)) 
    \__do_out[15]_i_8 
       (.I0(addr[3]),
        .I1(addr[5]),
        .I2(addr[2]),
        .I3(addr[4]),
        .I4(addr[1]),
        .I5(\__do_out[7]_i_5_n_0 ),
        .O(\__do_out[15]_i_8_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000010000)) 
    \__do_out[15]_i_9 
       (.I0(\__do_out[15]_i_16_n_0 ),
        .I1(addr[3]),
        .I2(addr[5]),
        .I3(addr[2]),
        .I4(addr[4]),
        .I5(addr[1]),
        .O(\__do_out[15]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hDDDDDDDDDDDDD0DD)) 
    \__do_out[1]_i_1 
       (.I0(\__do_out[15]_i_3_n_0 ),
        .I1(scratch[1]),
        .I2(\__do_out[1]_i_2_n_0 ),
        .I3(\__do_out[1]_i_3_n_0 ),
        .I4(\__do_out[1]_i_4_n_0 ),
        .I5(\__do_out[1]_i_5_n_0 ),
        .O(p_1_in__0[1]));
  LUT6 #(
    .INIT(64'hA0ACACA000000000)) 
    \__do_out[1]_i_10 
       (.I0(\gpio_dato_reg_n_0_[2] ),
        .I1(\rfmod_out[9]_INST_0_i_3_n_0 ),
        .I2(p_1_in),
        .I3(\u_tester/u_gen/out_reg_n_0_[1] ),
        .I4(test_stat[23]),
        .I5(\__do_out[15]_i_10_n_0 ),
        .O(\__do_out[1]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'hEFFFEFEFAAAAAAAA)) 
    \__do_out[1]_i_11 
       (.I0(\__do_out[1]_i_15_n_0 ),
        .I1(\__do_out[1]_i_16_n_0 ),
        .I2(\__do_out[1]_i_17_n_0 ),
        .I3(\__do_out[15]_i_19_n_0 ),
        .I4(\ctrl3_reg_n_0_[1] ),
        .I5(\__do_out[15]_i_20_n_0 ),
        .O(\__do_out[1]_i_11_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair142" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[1]_i_12 
       (.I0(i2c_data_o1[1]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[1]_i_12_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair179" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \__do_out[1]_i_13 
       (.I0(\jtag_ctrl_reg_n_0_[1] ),
        .I1(\__do_out[4]_i_13_n_0 ),
        .O(\__do_out[1]_i_13_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair169" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \__do_out[1]_i_14 
       (.I0(i2c_data_o0[1]),
        .I1(\__do_out[14]_i_19_n_0 ),
        .O(\__do_out[1]_i_14_n_0 ));
  LUT6 #(
    .INIT(64'h0008000C00080000)) 
    \__do_out[1]_i_15 
       (.I0(rfmod_in[2]),
        .I1(en),
        .I2(wr),
        .I3(\__do_out[15]_i_30_n_0 ),
        .I4(addr[0]),
        .I5(rfmod_id[1]),
        .O(\__do_out[1]_i_15_n_0 ));
  LUT5 #(
    .INIT(32'h00000020)) 
    \__do_out[1]_i_16 
       (.I0(ctrl2_rd[1]),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[1]_i_16_n_0 ));
  LUT6 #(
    .INIT(64'hFFCDFFFFFFFDFFFF)) 
    \__do_out[1]_i_17 
       (.I0(ctrl0_rd[1]),
        .I1(\__do_out[15]_i_28_n_0 ),
        .I2(addr[0]),
        .I3(wr),
        .I4(en),
        .I5(ctrl1_rd[1]),
        .O(\__do_out[1]_i_17_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair90" *) 
  LUT4 #(
    .INIT(16'hFEEE)) 
    \__do_out[1]_i_2 
       (.I0(\__do_out[15]_i_8_n_0 ),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[7]_i_2_n_0 ),
        .I3(gpio_exp_rb[17]),
        .O(\__do_out[1]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h5555F75555555555)) 
    \__do_out[1]_i_3 
       (.I0(\__do_out[14]_i_5_n_0 ),
        .I1(\__do_out[14]_i_9_n_0 ),
        .I2(\__do_out[1]_i_6_n_0 ),
        .I3(\__do_out[1]_i_7_n_0 ),
        .I4(\__do_out[1]_i_8_n_0 ),
        .I5(\__do_out[1]_i_9_n_0 ),
        .O(\__do_out[1]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair51" *) 
  LUT5 #(
    .INIT(32'hFEAAFAAA)) 
    \__do_out[1]_i_4 
       (.I0(\__do_out[1]_i_10_n_0 ),
        .I1(test_stat__0[1]),
        .I2(\__do_out[1]_i_11_n_0 ),
        .I3(rst),
        .I4(\__do_out[15]_i_12_n_0 ),
        .O(\__do_out[1]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair164" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \__do_out[1]_i_5 
       (.I0(gpio_exp_rb[1]),
        .I1(\__do_out[14]_i_6_n_0 ),
        .O(\__do_out[1]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000007)) 
    \__do_out[1]_i_6 
       (.I0(\__do_out[13]_i_11_n_0 ),
        .I1(test_stat__0[17]),
        .I2(\__do_out[1]_i_12_n_0 ),
        .I3(\__do_out[1]_i_13_n_0 ),
        .I4(\__do_out[1]_i_11_n_0 ),
        .I5(\__do_out[1]_i_14_n_0 ),
        .O(\__do_out[1]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair51" *) 
  LUT4 #(
    .INIT(16'h0777)) 
    \__do_out[1]_i_7 
       (.I0(\__do_out[1]_i_11_n_0 ),
        .I1(rst),
        .I2(\__do_out[15]_i_12_n_0 ),
        .I3(test_stat__0[1]),
        .O(\__do_out[1]_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair99" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[1]_i_8 
       (.I0(rfmod_in[2]),
        .I1(\__do_out[15]_i_14_n_0 ),
        .O(\__do_out[1]_i_8_n_0 ));
  LUT6 #(
    .INIT(64'h45FF4555FFFFFFFF)) 
    \__do_out[1]_i_9 
       (.I0(test_en),
        .I1(rst),
        .I2(dut_pgood),
        .I3(p_1_in),
        .I4(test_dir),
        .I5(\__do_out[15]_i_15_n_0 ),
        .O(\__do_out[1]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hDDDDDDDDDDD0DDDD)) 
    \__do_out[2]_i_1 
       (.I0(\__do_out[15]_i_3_n_0 ),
        .I1(scratch[2]),
        .I2(\__do_out[2]_i_2_n_0 ),
        .I3(\__do_out[2]_i_3_n_0 ),
        .I4(\__do_out[2]_i_4_n_0 ),
        .I5(\__do_out[2]_i_5_n_0 ),
        .O(p_1_in__0[2]));
  LUT6 #(
    .INIT(64'h45FF4555FFFFFFFF)) 
    \__do_out[2]_i_10 
       (.I0(test_en),
        .I1(rst),
        .I2(dut_pgood),
        .I3(\gpio_dir_reg_n_0_[3] ),
        .I4(test_dir),
        .I5(\__do_out[15]_i_15_n_0 ),
        .O(\__do_out[2]_i_10_n_0 ));
  LUT5 #(
    .INIT(32'h00000020)) 
    \__do_out[2]_i_11 
       (.I0(ctrl2_rd[2]),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[2]_i_11_n_0 ));
  LUT6 #(
    .INIT(64'hFFCDFFFFFFFDFFFF)) 
    \__do_out[2]_i_12 
       (.I0(ctrl0_rd[2]),
        .I1(\__do_out[15]_i_28_n_0 ),
        .I2(addr[0]),
        .I3(wr),
        .I4(en),
        .I5(ctrl1_rd[2]),
        .O(\__do_out[2]_i_12_n_0 ));
  LUT6 #(
    .INIT(64'h0008000C00080000)) 
    \__do_out[2]_i_13 
       (.I0(rfmod_in[3]),
        .I1(en),
        .I2(wr),
        .I3(\__do_out[15]_i_30_n_0 ),
        .I4(addr[0]),
        .I5(rfmod_id[2]),
        .O(\__do_out[2]_i_13_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair95" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[2]_i_14 
       (.I0(test_stat__0[18]),
        .I1(\__do_out[13]_i_11_n_0 ),
        .O(\__do_out[2]_i_14_n_0 ));
  LUT2 #(
    .INIT(4'h2)) 
    \__do_out[2]_i_15 
       (.I0(i2c_data_o0[2]),
        .I1(\__do_out[14]_i_19_n_0 ),
        .O(\__do_out[2]_i_15_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair179" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \__do_out[2]_i_16 
       (.I0(prog_jdi),
        .I1(\__do_out[4]_i_13_n_0 ),
        .O(\__do_out[2]_i_16_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair142" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[2]_i_17 
       (.I0(i2c_data_o1[2]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[2]_i_17_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair90" *) 
  LUT4 #(
    .INIT(16'hFEEE)) 
    \__do_out[2]_i_2 
       (.I0(\__do_out[15]_i_8_n_0 ),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[7]_i_2_n_0 ),
        .I3(gpio_exp_rb[18]),
        .O(\__do_out[2]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair52" *) 
  LUT5 #(
    .INIT(32'hFFFFA222)) 
    \__do_out[2]_i_3 
       (.I0(rst),
        .I1(\__do_out[2]_i_6_n_0 ),
        .I2(test_stat__0[2]),
        .I3(\__do_out[15]_i_12_n_0 ),
        .I4(\__do_out[2]_i_7_n_0 ),
        .O(\__do_out[2]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair165" *) 
  LUT2 #(
    .INIT(4'hB)) 
    \__do_out[2]_i_4 
       (.I0(\__do_out[14]_i_6_n_0 ),
        .I1(gpio_exp_rb[2]),
        .O(\__do_out[2]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAA80AAAAAAAAAAAA)) 
    \__do_out[2]_i_5 
       (.I0(\__do_out[14]_i_5_n_0 ),
        .I1(\__do_out[15]_i_14_n_0 ),
        .I2(rfmod_in[3]),
        .I3(\__do_out[2]_i_8_n_0 ),
        .I4(\__do_out[2]_i_9_n_0 ),
        .I5(\__do_out[2]_i_10_n_0 ),
        .O(\__do_out[2]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h0000000075557575)) 
    \__do_out[2]_i_6 
       (.I0(\__do_out[15]_i_20_n_0 ),
        .I1(\__do_out[2]_i_11_n_0 ),
        .I2(\__do_out[2]_i_12_n_0 ),
        .I3(\__do_out[15]_i_19_n_0 ),
        .I4(\ctrl3_reg_n_0_[2] ),
        .I5(\__do_out[2]_i_13_n_0 ),
        .O(\__do_out[2]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'hA0ACACA000000000)) 
    \__do_out[2]_i_7 
       (.I0(\gpio_dato_reg_n_0_[3] ),
        .I1(\rfmod_out[9]_INST_0_i_3_n_0 ),
        .I2(\gpio_dir_reg_n_0_[3] ),
        .I3(\u_tester/u_gen/out_reg_n_0_[2] ),
        .I4(test_stat[23]),
        .I5(\__do_out[15]_i_10_n_0 ),
        .O(\__do_out[2]_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair52" *) 
  LUT4 #(
    .INIT(16'h8F88)) 
    \__do_out[2]_i_8 
       (.I0(\__do_out[15]_i_12_n_0 ),
        .I1(test_stat__0[2]),
        .I2(\__do_out[2]_i_6_n_0 ),
        .I3(rst),
        .O(\__do_out[2]_i_8_n_0 ));
  LUT6 #(
    .INIT(64'h00000010FFFFFFFF)) 
    \__do_out[2]_i_9 
       (.I0(\__do_out[2]_i_14_n_0 ),
        .I1(\__do_out[2]_i_15_n_0 ),
        .I2(\__do_out[2]_i_6_n_0 ),
        .I3(\__do_out[2]_i_16_n_0 ),
        .I4(\__do_out[2]_i_17_n_0 ),
        .I5(\__do_out[14]_i_9_n_0 ),
        .O(\__do_out[2]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hDDDDDDDDDDD0DDDD)) 
    \__do_out[3]_i_1 
       (.I0(\__do_out[15]_i_3_n_0 ),
        .I1(scratch[3]),
        .I2(\__do_out[3]_i_2_n_0 ),
        .I3(\__do_out[3]_i_3_n_0 ),
        .I4(\__do_out[3]_i_4_n_0 ),
        .I5(\__do_out[3]_i_5_n_0 ),
        .O(p_1_in__0[3]));
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[3]_i_10 
       (.I0(rfmod_in[4]),
        .I1(\__do_out[15]_i_14_n_0 ),
        .O(\__do_out[3]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'h45FF4555FFFFFFFF)) 
    \__do_out[3]_i_11 
       (.I0(test_en),
        .I1(rst),
        .I2(dut_pgood),
        .I3(\gpio_dir_reg_n_0_[4] ),
        .I4(test_dir),
        .I5(\__do_out[15]_i_15_n_0 ),
        .O(\__do_out[3]_i_11_n_0 ));
  LUT6 #(
    .INIT(64'h0008000C00080000)) 
    \__do_out[3]_i_12 
       (.I0(rfmod_in[4]),
        .I1(en),
        .I2(wr),
        .I3(\__do_out[15]_i_30_n_0 ),
        .I4(addr[0]),
        .I5(rfmod_id[3]),
        .O(\__do_out[3]_i_12_n_0 ));
  LUT5 #(
    .INIT(32'h00000020)) 
    \__do_out[3]_i_13 
       (.I0(ctrl2_rd[3]),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[3]_i_13_n_0 ));
  LUT6 #(
    .INIT(64'hFDFFFCFFFDFFFFFF)) 
    \__do_out[3]_i_14 
       (.I0(ctrl1_rd[3]),
        .I1(\__do_out[15]_i_28_n_0 ),
        .I2(wr),
        .I3(en),
        .I4(addr[0]),
        .I5(ctrl0_rd[3]),
        .O(\__do_out[3]_i_14_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair171" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[3]_i_15 
       (.I0(i2c_data_o1[3]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[3]_i_15_n_0 ));
  LUT5 #(
    .INIT(32'h0000F200)) 
    \__do_out[3]_i_16 
       (.I0(dut_ten),
        .I1(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I2(prog_jen),
        .I3(rfmod_in[9]),
        .I4(\__do_out[4]_i_13_n_0 ),
        .O(\__do_out[3]_i_16_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair180" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[3]_i_17 
       (.I0(test_stat__0[19]),
        .I1(\__do_out[13]_i_11_n_0 ),
        .O(\__do_out[3]_i_17_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair93" *) 
  LUT4 #(
    .INIT(16'hFEEE)) 
    \__do_out[3]_i_2 
       (.I0(\__do_out[15]_i_8_n_0 ),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[7]_i_2_n_0 ),
        .I3(gpio_exp_rb[19]),
        .O(\__do_out[3]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair149" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[3]_i_3 
       (.I0(gpio_exp_rb[3]),
        .I1(\__do_out[15]_i_9_n_0 ),
        .O(\__do_out[3]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair49" *) 
  LUT5 #(
    .INIT(32'h00001F5F)) 
    \__do_out[3]_i_4 
       (.I0(\__do_out[3]_i_6_n_0 ),
        .I1(test_stat__0[3]),
        .I2(rst),
        .I3(\__do_out[15]_i_12_n_0 ),
        .I4(\__do_out[3]_i_7_n_0 ),
        .O(\__do_out[3]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAAA20AAAAAAAA)) 
    \__do_out[3]_i_5 
       (.I0(\__do_out[14]_i_5_n_0 ),
        .I1(\__do_out[3]_i_8_n_0 ),
        .I2(\__do_out[14]_i_9_n_0 ),
        .I3(\__do_out[3]_i_9_n_0 ),
        .I4(\__do_out[3]_i_10_n_0 ),
        .I5(\__do_out[3]_i_11_n_0 ),
        .O(\__do_out[3]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hEFFFEFEFAAAAAAAA)) 
    \__do_out[3]_i_6 
       (.I0(\__do_out[3]_i_12_n_0 ),
        .I1(\__do_out[3]_i_13_n_0 ),
        .I2(\__do_out[3]_i_14_n_0 ),
        .I3(\__do_out[15]_i_19_n_0 ),
        .I4(\ctrl3_reg_n_0_[3] ),
        .I5(\__do_out[15]_i_20_n_0 ),
        .O(\__do_out[3]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'hA0ACACA000000000)) 
    \__do_out[3]_i_7 
       (.I0(\gpio_dato_reg_n_0_[4] ),
        .I1(\rfmod_out[9]_INST_0_i_3_n_0 ),
        .I2(\gpio_dir_reg_n_0_[4] ),
        .I3(test_stat[23]),
        .I4(\u_tester/u_gen/out_reg_n_0_[3] ),
        .I5(\__do_out[15]_i_10_n_0 ),
        .O(\__do_out[3]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'h0000000001000101)) 
    \__do_out[3]_i_8 
       (.I0(\__do_out[3]_i_15_n_0 ),
        .I1(\__do_out[3]_i_6_n_0 ),
        .I2(\__do_out[3]_i_16_n_0 ),
        .I3(\__do_out[14]_i_19_n_0 ),
        .I4(i2c_data_o0[3]),
        .I5(\__do_out[3]_i_17_n_0 ),
        .O(\__do_out[3]_i_8_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair49" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \__do_out[3]_i_9 
       (.I0(\__do_out[15]_i_12_n_0 ),
        .I1(test_stat__0[3]),
        .I2(rst),
        .I3(\__do_out[3]_i_6_n_0 ),
        .O(\__do_out[3]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hA8AAAAAAA8AAA8AA)) 
    \__do_out[4]_i_1 
       (.I0(\__do_out[4]_i_2_n_0 ),
        .I1(\__do_out[4]_i_3_n_0 ),
        .I2(\__do_out[4]_i_4_n_0 ),
        .I3(\__do_out[4]_i_5_n_0 ),
        .I4(\__do_out[4]_i_6_n_0 ),
        .I5(\__do_out[14]_i_5_n_0 ),
        .O(p_1_in__0[4]));
  (* SOFT_HLUTNM = "soft_lutpair95" *) 
  LUT4 #(
    .INIT(16'h5700)) 
    \__do_out[4]_i_10 
       (.I0(\st_cur[0]_i_2_n_0 ),
        .I1(\u_tester/st_cur [1]),
        .I2(\u_tester/st_cur [0]),
        .I3(\__do_out[13]_i_11_n_0 ),
        .O(\__do_out[4]_i_10_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair175" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[4]_i_11 
       (.I0(rfmod_in[5]),
        .I1(\__do_out[15]_i_14_n_0 ),
        .O(\__do_out[4]_i_11_n_0 ));
  LUT6 #(
    .INIT(64'h0000000075557575)) 
    \__do_out[4]_i_12 
       (.I0(\__do_out[15]_i_20_n_0 ),
        .I1(\__do_out[4]_i_15_n_0 ),
        .I2(\__do_out[4]_i_16_n_0 ),
        .I3(\__do_out[15]_i_19_n_0 ),
        .I4(\ctrl3_reg_n_0_[4] ),
        .I5(\__do_out[4]_i_17_n_0 ),
        .O(\__do_out[4]_i_12_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFEFFFFFFFFFF)) 
    \__do_out[4]_i_13 
       (.I0(\__do_out[7]_i_5_n_0 ),
        .I1(addr[5]),
        .I2(addr[3]),
        .I3(addr[2]),
        .I4(addr[4]),
        .I5(addr[1]),
        .O(\__do_out[4]_i_13_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair171" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[4]_i_14 
       (.I0(i2c_data_o1[4]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[4]_i_14_n_0 ));
  LUT5 #(
    .INIT(32'h00000020)) 
    \__do_out[4]_i_15 
       (.I0(ctrl2_rd[4]),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[4]_i_15_n_0 ));
  LUT6 #(
    .INIT(64'hFFCDFFFFFFFDFFFF)) 
    \__do_out[4]_i_16 
       (.I0(ctrl0_rd[4]),
        .I1(\__do_out[15]_i_28_n_0 ),
        .I2(addr[0]),
        .I3(wr),
        .I4(en),
        .I5(ctrl1_rd[4]),
        .O(\__do_out[4]_i_16_n_0 ));
  LUT6 #(
    .INIT(64'h000800000008000C)) 
    \__do_out[4]_i_17 
       (.I0(rfmod_in[5]),
        .I1(en),
        .I2(wr),
        .I3(\__do_out[15]_i_30_n_0 ),
        .I4(addr[0]),
        .I5(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .O(\__do_out[4]_i_17_n_0 ));
  LUT2 #(
    .INIT(4'hB)) 
    \__do_out[4]_i_2 
       (.I0(scratch[4]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .O(\__do_out[4]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair93" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \__do_out[4]_i_3 
       (.I0(\__do_out[15]_i_3_n_0 ),
        .I1(\__do_out[7]_i_2_n_0 ),
        .I2(gpio_exp_rb[20]),
        .O(\__do_out[4]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair87" *) 
  LUT4 #(
    .INIT(16'hF888)) 
    \__do_out[4]_i_4 
       (.I0(rfmod_out[5]),
        .I1(\__do_out[15]_i_10_n_0 ),
        .I2(rst),
        .I3(\__do_out[4]_i_7_n_0 ),
        .O(\__do_out[4]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair165" *) 
  LUT2 #(
    .INIT(4'hB)) 
    \__do_out[4]_i_5 
       (.I0(\__do_out[14]_i_6_n_0 ),
        .I1(gpio_exp_rb[4]),
        .O(\__do_out[4]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h00000000000022A2)) 
    \__do_out[4]_i_6 
       (.I0(\__do_out[4]_i_8_n_0 ),
        .I1(\__do_out[14]_i_9_n_0 ),
        .I2(\__do_out[4]_i_9_n_0 ),
        .I3(\__do_out[4]_i_10_n_0 ),
        .I4(\__do_out[4]_i_7_n_0 ),
        .I5(\__do_out[4]_i_11_n_0 ),
        .O(\__do_out[4]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair73" *) 
  LUT4 #(
    .INIT(16'h8F88)) 
    \__do_out[4]_i_7 
       (.I0(\__do_out[15]_i_12_n_0 ),
        .I1(test_stat__0[4]),
        .I2(\__do_out[4]_i_12_n_0 ),
        .I3(rst),
        .O(\__do_out[4]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'hFFFF001DFFFFFFFF)) 
    \__do_out[4]_i_8 
       (.I0(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I1(test_en),
        .I2(test_dir),
        .I3(p_2_in),
        .I4(\rfmod_oe[13]_INST_0_i_1_n_0 ),
        .I5(\__do_out[15]_i_15_n_0 ),
        .O(\__do_out[4]_i_8_n_0 ));
  LUT6 #(
    .INIT(64'h00000000D000D0D0)) 
    \__do_out[4]_i_9 
       (.I0(i2c_data_o0[4]),
        .I1(\__do_out[14]_i_19_n_0 ),
        .I2(\__do_out[4]_i_12_n_0 ),
        .I3(\__do_out[4]_i_13_n_0 ),
        .I4(prog_jen),
        .I5(\__do_out[4]_i_14_n_0 ),
        .O(\__do_out[4]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hBBBBBBBBB888BBBB)) 
    \__do_out[5]_i_1 
       (.I0(scratch[5]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[7]_i_2_n_0 ),
        .I3(gpio_exp_rb[21]),
        .I4(\__do_out[5]_i_2_n_0 ),
        .I5(\__do_out[5]_i_3_n_0 ),
        .O(p_1_in__0[5]));
  LUT6 #(
    .INIT(64'hDD0DDD0D0000DD0D)) 
    \__do_out[5]_i_10 
       (.I0(ctrl2_rd[5]),
        .I1(\__do_out[13]_i_15_n_0 ),
        .I2(ctrl1_rd[5]),
        .I3(\__do_out[13]_i_14_n_0 ),
        .I4(ctrl0_rd[5]),
        .I5(\__do_out[13]_i_13_n_0 ),
        .O(\__do_out[5]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAAAAAC000030C)) 
    \__do_out[5]_i_11 
       (.I0(test_dir),
        .I1(rfmod_id[1]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[0]),
        .I4(rfmod_id[3]),
        .I5(test_en),
        .O(\__do_out[5]_i_11_n_0 ));
  LUT6 #(
    .INIT(64'h0000000007007777)) 
    \__do_out[5]_i_2 
       (.I0(rfmod_out[6]),
        .I1(\__do_out[15]_i_10_n_0 ),
        .I2(\__do_out[5]_i_4_n_0 ),
        .I3(\__do_out[5]_i_5_n_0 ),
        .I4(rst),
        .I5(\__do_out[5]_i_6_n_0 ),
        .O(\__do_out[5]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAA8A8A8AAAAAAAAA)) 
    \__do_out[5]_i_3 
       (.I0(\__do_out[14]_i_5_n_0 ),
        .I1(\__do_out[5]_i_4_n_0 ),
        .I2(\__do_out[5]_i_5_n_0 ),
        .I3(\__do_out[15]_i_14_n_0 ),
        .I4(rfmod_in[6]),
        .I5(\__do_out[5]_i_7_n_0 ),
        .O(\__do_out[5]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hAA00AAFCAA00AA30)) 
    \__do_out[5]_i_4 
       (.I0(test_stat__0[5]),
        .I1(\__do_out[13]_i_11_n_0 ),
        .I2(\__do_out[5]_i_8_n_0 ),
        .I3(\__do_out[15]_i_12_n_0 ),
        .I4(rst),
        .I5(test_stat[21]),
        .O(\__do_out[5]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'h8088AAAA)) 
    \__do_out[5]_i_5 
       (.I0(\__do_out[5]_i_9_n_0 ),
        .I1(\__do_out[5]_i_10_n_0 ),
        .I2(\__do_out[15]_i_19_n_0 ),
        .I3(\ctrl3_reg_n_0_[5] ),
        .I4(\__do_out[15]_i_20_n_0 ),
        .O(\__do_out[5]_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair161" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[5]_i_6 
       (.I0(gpio_exp_rb[5]),
        .I1(\__do_out[15]_i_9_n_0 ),
        .O(\__do_out[5]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'h1F111F1FFFFFFFFF)) 
    \__do_out[5]_i_7 
       (.I0(\gpio_dir_reg_n_0_[6] ),
        .I1(\__do_out[5]_i_11_n_0 ),
        .I2(test_en),
        .I3(rst),
        .I4(dut_pgood),
        .I5(\__do_out[15]_i_15_n_0 ),
        .O(\__do_out[5]_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair57" *) 
  LUT5 #(
    .INIT(32'h0000F202)) 
    \__do_out[5]_i_8 
       (.I0(i2c_data_o0[5]),
        .I1(\__do_out[14]_i_19_n_0 ),
        .I2(\__do_out[14]_i_17_n_0 ),
        .I3(i2c_data_o1[5]),
        .I4(\idovr[4]_i_2_n_0 ),
        .O(\__do_out[5]_i_8_n_0 ));
  LUT4 #(
    .INIT(16'hD0DD)) 
    \__do_out[5]_i_9 
       (.I0(rfmod_in[6]),
        .I1(\__do_out[13]_i_16_n_0 ),
        .I2(\__do_out[13]_i_17_n_0 ),
        .I3(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .O(\__do_out[5]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hBBBBBBBBB888BBBB)) 
    \__do_out[6]_i_1 
       (.I0(scratch[6]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[7]_i_2_n_0 ),
        .I3(gpio_exp_rb[22]),
        .I4(\__do_out[6]_i_2_n_0 ),
        .I5(\__do_out[6]_i_3_n_0 ),
        .O(p_1_in__0[6]));
  LUT6 #(
    .INIT(64'h0003000200000002)) 
    \__do_out[6]_i_10 
       (.I0(\rfmod_oe[8]_INST_0_i_1_n_0 ),
        .I1(addr[1]),
        .I2(\__do_out[0]_i_6_n_0 ),
        .I3(\__do_out[15]_i_29_n_0 ),
        .I4(addr[0]),
        .I5(rfmod_in[7]),
        .O(\__do_out[6]_i_10_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair36" *) 
  LUT5 #(
    .INIT(32'h00000800)) 
    \__do_out[6]_i_11 
       (.I0(\ctrl3_reg_n_0_[6] ),
        .I1(en),
        .I2(wr),
        .I3(addr[0]),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[6]_i_11_n_0 ));
  LUT6 #(
    .INIT(64'hFDFFFCFFFDFFFFFF)) 
    \__do_out[6]_i_12 
       (.I0(ctrl1_rd[6]),
        .I1(\__do_out[15]_i_28_n_0 ),
        .I2(wr),
        .I3(en),
        .I4(addr[0]),
        .I5(ctrl0_rd[6]),
        .O(\__do_out[6]_i_12_n_0 ));
  LUT5 #(
    .INIT(32'h00000020)) 
    \__do_out[6]_i_13 
       (.I0(ctrl2_rd[6]),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[6]_i_13_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000020)) 
    \__do_out[6]_i_14 
       (.I0(i2c_data_o0[6]),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(addr[1]),
        .I5(\__do_out[15]_i_35_n_0 ),
        .O(\__do_out[6]_i_14_n_0 ));
  LUT6 #(
    .INIT(64'h00000DDD0DDD0DDD)) 
    \__do_out[6]_i_2 
       (.I0(gpio_exp_rb[6]),
        .I1(\__do_out[14]_i_6_n_0 ),
        .I2(rfmod_out[7]),
        .I3(\__do_out[15]_i_10_n_0 ),
        .I4(\__do_out[6]_i_4_n_0 ),
        .I5(rst),
        .O(\__do_out[6]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAA888A888A888)) 
    \__do_out[6]_i_3 
       (.I0(\__do_out[14]_i_5_n_0 ),
        .I1(\__do_out[6]_i_4_n_0 ),
        .I2(\__do_out[15]_i_14_n_0 ),
        .I3(rfmod_in[7]),
        .I4(rfmod_oe[7]),
        .I5(\__do_out[15]_i_15_n_0 ),
        .O(\__do_out[6]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hFEFEFEFEFEEEFEFE)) 
    \__do_out[6]_i_4 
       (.I0(\__do_out[6]_i_5_n_0 ),
        .I1(\__do_out[6]_i_6_n_0 ),
        .I2(\__do_out[14]_i_9_n_0 ),
        .I3(\__do_out[6]_i_7_n_0 ),
        .I4(\__do_out[6]_i_8_n_0 ),
        .I5(\__do_out[6]_i_9_n_0 ),
        .O(\__do_out[6]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair73" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[6]_i_5 
       (.I0(test_stat__0[6]),
        .I1(\__do_out[15]_i_12_n_0 ),
        .O(\__do_out[6]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAA8AA88888888)) 
    \__do_out[6]_i_6 
       (.I0(\__do_out[15]_i_32_n_0 ),
        .I1(\__do_out[6]_i_10_n_0 ),
        .I2(\__do_out[6]_i_11_n_0 ),
        .I3(\__do_out[6]_i_12_n_0 ),
        .I4(\__do_out[6]_i_13_n_0 ),
        .I5(\__do_out[15]_i_20_n_0 ),
        .O(\__do_out[6]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair141" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[6]_i_7 
       (.I0(i2c_data_o1[6]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[6]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000105555)) 
    \__do_out[6]_i_8 
       (.I0(\__do_out[6]_i_10_n_0 ),
        .I1(\__do_out[6]_i_11_n_0 ),
        .I2(\__do_out[6]_i_12_n_0 ),
        .I3(\__do_out[6]_i_13_n_0 ),
        .I4(\__do_out[15]_i_20_n_0 ),
        .I5(\__do_out[6]_i_14_n_0 ),
        .O(\__do_out[6]_i_8_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair182" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[6]_i_9 
       (.I0(test_stat[22]),
        .I1(\__do_out[13]_i_11_n_0 ),
        .O(\__do_out[6]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hBBBBBBBBB888BBBB)) 
    \__do_out[7]_i_1 
       (.I0(scratch[7]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[7]_i_2_n_0 ),
        .I3(gpio_exp_rb[23]),
        .I4(\__do_out[7]_i_3_n_0 ),
        .I5(\__do_out[7]_i_4_n_0 ),
        .O(p_1_in__0[7]));
  LUT6 #(
    .INIT(64'hAAAAA8AA88888888)) 
    \__do_out[7]_i_10 
       (.I0(rst),
        .I1(\__do_out[7]_i_12_n_0 ),
        .I2(\__do_out[7]_i_13_n_0 ),
        .I3(\__do_out[7]_i_14_n_0 ),
        .I4(\__do_out[7]_i_15_n_0 ),
        .I5(\__do_out[15]_i_20_n_0 ),
        .O(\__do_out[7]_i_10_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair177" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[7]_i_11 
       (.I0(test_stat__0[7]),
        .I1(\__do_out[15]_i_12_n_0 ),
        .O(\__do_out[7]_i_11_n_0 ));
  LUT6 #(
    .INIT(64'h0008000C00080000)) 
    \__do_out[7]_i_12 
       (.I0(rfmod_in[8]),
        .I1(en),
        .I2(wr),
        .I3(\__do_out[15]_i_30_n_0 ),
        .I4(addr[0]),
        .I5(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .O(\__do_out[7]_i_12_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT5 #(
    .INIT(32'h00000800)) 
    \__do_out[7]_i_13 
       (.I0(\ctrl3_reg_n_0_[7] ),
        .I1(en),
        .I2(wr),
        .I3(addr[0]),
        .I4(\__do_out[15]_i_27_n_0 ),
        .O(\__do_out[7]_i_13_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT5 #(
    .INIT(32'hFFFBFFFF)) 
    \__do_out[7]_i_14 
       (.I0(addr[0]),
        .I1(en),
        .I2(wr),
        .I3(\__do_out[15]_i_27_n_0 ),
        .I4(ctrl2_rd[7]),
        .O(\__do_out[7]_i_14_n_0 ));
  LUT6 #(
    .INIT(64'h0200030002000000)) 
    \__do_out[7]_i_15 
       (.I0(ctrl1_rd[7]),
        .I1(\__do_out[15]_i_28_n_0 ),
        .I2(wr),
        .I3(en),
        .I4(addr[0]),
        .I5(ctrl0_rd[7]),
        .O(\__do_out[7]_i_15_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000020)) 
    \__do_out[7]_i_16 
       (.I0(i2c_data_o0[7]),
        .I1(addr[0]),
        .I2(en),
        .I3(wr),
        .I4(addr[1]),
        .I5(\__do_out[15]_i_35_n_0 ),
        .O(\__do_out[7]_i_16_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000010000)) 
    \__do_out[7]_i_2 
       (.I0(\__do_out[7]_i_5_n_0 ),
        .I1(addr[3]),
        .I2(addr[5]),
        .I3(addr[2]),
        .I4(addr[4]),
        .I5(addr[1]),
        .O(\__do_out[7]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0DDD0DDD00000DDD)) 
    \__do_out[7]_i_3 
       (.I0(gpio_exp_rb[7]),
        .I1(\__do_out[14]_i_6_n_0 ),
        .I2(rfmod_out[8]),
        .I3(\__do_out[15]_i_10_n_0 ),
        .I4(rst),
        .I5(\__do_out[7]_i_6_n_0 ),
        .O(\__do_out[7]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAA222A222A222)) 
    \__do_out[7]_i_4 
       (.I0(\__do_out[14]_i_5_n_0 ),
        .I1(\__do_out[7]_i_6_n_0 ),
        .I2(\__do_out[15]_i_14_n_0 ),
        .I3(rfmod_in[8]),
        .I4(rfmod_oe[8]),
        .I5(\__do_out[15]_i_15_n_0 ),
        .O(\__do_out[7]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair38" *) 
  LUT3 #(
    .INIT(8'hDF)) 
    \__do_out[7]_i_5 
       (.I0(en),
        .I1(wr),
        .I2(addr[0]),
        .O(\__do_out[7]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000005575)) 
    \__do_out[7]_i_6 
       (.I0(\__do_out[14]_i_9_n_0 ),
        .I1(\__do_out[7]_i_7_n_0 ),
        .I2(\__do_out[7]_i_8_n_0 ),
        .I3(\__do_out[7]_i_9_n_0 ),
        .I4(\__do_out[7]_i_10_n_0 ),
        .I5(\__do_out[7]_i_11_n_0 ),
        .O(\__do_out[7]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair181" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[7]_i_7 
       (.I0(i2c_data_o1[7]),
        .I1(\__do_out[14]_i_17_n_0 ),
        .O(\__do_out[7]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000105555)) 
    \__do_out[7]_i_8 
       (.I0(\__do_out[7]_i_12_n_0 ),
        .I1(\__do_out[7]_i_13_n_0 ),
        .I2(\__do_out[7]_i_14_n_0 ),
        .I3(\__do_out[7]_i_15_n_0 ),
        .I4(\__do_out[15]_i_20_n_0 ),
        .I5(\__do_out[7]_i_16_n_0 ),
        .O(\__do_out[7]_i_8_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair156" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[7]_i_9 
       (.I0(test_stat[23]),
        .I1(\__do_out[13]_i_11_n_0 ),
        .O(\__do_out[7]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hB8BBBBBBB8BBB8BB)) 
    \__do_out[8]_i_1 
       (.I0(scratch[8]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[8]_i_2_n_0 ),
        .I3(\__do_out[8]_i_3_n_0 ),
        .I4(\__do_out[8]_i_4_n_0 ),
        .I5(\__do_out[14]_i_5_n_0 ),
        .O(p_1_in__0[8]));
  (* SOFT_HLUTNM = "soft_lutpair91" *) 
  LUT4 #(
    .INIT(16'h4F44)) 
    \__do_out[8]_i_10 
       (.I0(\__do_out[13]_i_16_n_0 ),
        .I1(rfmod_in[9]),
        .I2(\__do_out[13]_i_17_n_0 ),
        .I3(tx_active[0]),
        .O(\__do_out[8]_i_10_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \__do_out[8]_i_2 
       (.I0(\__do_out[7]_i_2_n_0 ),
        .I1(gpio_exp_rb[24]),
        .I2(\__do_out[15]_i_8_n_0 ),
        .I3(p_29_in[8]),
        .O(\__do_out[8]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000007007777)) 
    \__do_out[8]_i_3 
       (.I0(rfmod_out[9]),
        .I1(\__do_out[15]_i_10_n_0 ),
        .I2(\__do_out[8]_i_5_n_0 ),
        .I3(\__do_out[8]_i_6_n_0 ),
        .I4(rst),
        .I5(\__do_out[8]_i_7_n_0 ),
        .O(\__do_out[8]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h0000000007770000)) 
    \__do_out[8]_i_4 
       (.I0(\__do_out[15]_i_15_n_0 ),
        .I1(rfmod_oe[9]),
        .I2(rfmod_in[9]),
        .I3(\__do_out[15]_i_14_n_0 ),
        .I4(\__do_out[8]_i_6_n_0 ),
        .I5(\__do_out[8]_i_5_n_0 ),
        .O(\__do_out[8]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFF8F8F888888888)) 
    \__do_out[8]_i_5 
       (.I0(\__do_out[15]_i_12_n_0 ),
        .I1(test_stat__0[8]),
        .I2(\__do_out[13]_i_11_n_0 ),
        .I3(\__do_out[14]_i_17_n_0 ),
        .I4(i2c_data_o0[0]),
        .I5(\__do_out[8]_i_8_n_0 ),
        .O(\__do_out[8]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h00008AFF)) 
    \__do_out[8]_i_6 
       (.I0(\__do_out[8]_i_9_n_0 ),
        .I1(\__do_out[15]_i_19_n_0 ),
        .I2(\ctrl3_reg_n_0_[8] ),
        .I3(\__do_out[15]_i_20_n_0 ),
        .I4(\__do_out[8]_i_10_n_0 ),
        .O(\__do_out[8]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair167" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[8]_i_7 
       (.I0(gpio_exp_rb[8]),
        .I1(\__do_out[15]_i_9_n_0 ),
        .O(\__do_out[8]_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair6" *) 
  LUT5 #(
    .INIT(32'h00011111)) 
    \__do_out[8]_i_8 
       (.I0(\__do_out[15]_i_12_n_0 ),
        .I1(rst),
        .I2(\u_tester/st_cur [1]),
        .I3(\u_tester/st_cur [0]),
        .I4(\__do_out[13]_i_11_n_0 ),
        .O(\__do_out[8]_i_8_n_0 ));
  LUT6 #(
    .INIT(64'hD0DD0000D0DDD0DD)) 
    \__do_out[8]_i_9 
       (.I0(ctrl0_rd[8]),
        .I1(\__do_out[13]_i_13_n_0 ),
        .I2(\__do_out[13]_i_14_n_0 ),
        .I3(ctrl1_rd[8]),
        .I4(\__do_out[13]_i_15_n_0 ),
        .I5(ctrl2_rd[8]),
        .O(\__do_out[8]_i_9_n_0 ));
  LUT6 #(
    .INIT(64'hB8BBBBBBB8BBB8BB)) 
    \__do_out[9]_i_1 
       (.I0(scratch[9]),
        .I1(\__do_out[15]_i_3_n_0 ),
        .I2(\__do_out[9]_i_2_n_0 ),
        .I3(\__do_out[9]_i_3_n_0 ),
        .I4(\__do_out[9]_i_4_n_0 ),
        .I5(\__do_out[14]_i_5_n_0 ),
        .O(p_1_in__0[9]));
  LUT6 #(
    .INIT(64'h22F2FFFF22F222F2)) 
    \__do_out[9]_i_10 
       (.I0(ctrl0_rd[9]),
        .I1(\__do_out[13]_i_13_n_0 ),
        .I2(ctrl1_rd[9]),
        .I3(\__do_out[13]_i_14_n_0 ),
        .I4(\__do_out[13]_i_15_n_0 ),
        .I5(ctrl2_rd[9]),
        .O(\__do_out[9]_i_10_n_0 ));
  LUT6 #(
    .INIT(64'h0000002300000020)) 
    \__do_out[9]_i_11 
       (.I0(rfmod_in[10]),
        .I1(\__do_out[15]_i_29_n_0 ),
        .I2(addr[0]),
        .I3(\__do_out[0]_i_6_n_0 ),
        .I4(addr[1]),
        .I5(tx_active[1]),
        .O(\__do_out[9]_i_11_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \__do_out[9]_i_2 
       (.I0(\__do_out[7]_i_2_n_0 ),
        .I1(gpio_exp_rb[25]),
        .I2(\__do_out[15]_i_8_n_0 ),
        .I3(p_29_in[9]),
        .O(\__do_out[9]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000007007777)) 
    \__do_out[9]_i_3 
       (.I0(rfmod_out[10]),
        .I1(\__do_out[15]_i_10_n_0 ),
        .I2(\__do_out[9]_i_5_n_0 ),
        .I3(\__do_out[9]_i_6_n_0 ),
        .I4(rst),
        .I5(\__do_out[9]_i_7_n_0 ),
        .O(\__do_out[9]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h000000000DDD0000)) 
    \__do_out[9]_i_4 
       (.I0(\__do_out[15]_i_15_n_0 ),
        .I1(\__do_out[9]_i_8_n_0 ),
        .I2(rfmod_in[10]),
        .I3(\__do_out[15]_i_14_n_0 ),
        .I4(\__do_out[9]_i_6_n_0 ),
        .I5(\__do_out[9]_i_5_n_0 ),
        .O(\__do_out[9]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hFFF8F8F888888888)) 
    \__do_out[9]_i_5 
       (.I0(\__do_out[15]_i_12_n_0 ),
        .I1(test_stat__0[9]),
        .I2(\__do_out[13]_i_11_n_0 ),
        .I3(\__do_out[14]_i_17_n_0 ),
        .I4(i2c_data_o0[1]),
        .I5(\__do_out[9]_i_9_n_0 ),
        .O(\__do_out[9]_i_5_n_0 ));
  LUT5 #(
    .INIT(32'h00001FBF)) 
    \__do_out[9]_i_6 
       (.I0(\__do_out[15]_i_19_n_0 ),
        .I1(\ctrl3_reg_n_0_[9] ),
        .I2(\__do_out[15]_i_20_n_0 ),
        .I3(\__do_out[9]_i_10_n_0 ),
        .I4(\__do_out[9]_i_11_n_0 ),
        .O(\__do_out[9]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair167" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \__do_out[9]_i_7 
       (.I0(gpio_exp_rb[9]),
        .I1(\__do_out[15]_i_9_n_0 ),
        .O(\__do_out[9]_i_7_n_0 ));
  LUT6 #(
    .INIT(64'h0F0FBF000000BF00)) 
    \__do_out[9]_i_8 
       (.I0(rst),
        .I1(dut_pgood),
        .I2(\gpio_dir_reg_n_0_[10] ),
        .I3(sda_t),
        .I4(test_en),
        .I5(test_dir),
        .O(\__do_out[9]_i_8_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair62" *) 
  LUT5 #(
    .INIT(32'h01110101)) 
    \__do_out[9]_i_9 
       (.I0(\__do_out[15]_i_12_n_0 ),
        .I1(rst),
        .I2(\__do_out[13]_i_11_n_0 ),
        .I3(\u_tester/st_cur [0]),
        .I4(\u_tester/st_cur [1]),
        .O(\__do_out[9]_i_9_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[0] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[0]),
        .Q(dato[0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[10] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[10]),
        .Q(dato[10]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[11] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[11]),
        .Q(dato[11]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[12] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[12]),
        .Q(dato[12]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[13] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[13]),
        .Q(dato[13]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[14] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[14]),
        .Q(dato[14]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[15] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[15]),
        .Q(dato[15]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[1] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[1]),
        .Q(dato[1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[2] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[2]),
        .Q(dato[2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[3] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[3]),
        .Q(dato[3]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[4] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[4]),
        .Q(dato[4]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[5] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[5]),
        .Q(dato[5]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[6] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[6]),
        .Q(dato[6]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[7] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[7]),
        .Q(dato[7]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[8] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[8]),
        .Q(dato[8]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \__do_out_reg[9] 
       (.C(clk),
        .CE(\__do_out[15]_i_1_n_0 ),
        .D(p_1_in__0[9]),
        .Q(dato[9]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b1)) 
    \__rdy_out_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(en),
        .Q(rdy),
        .R(rst));
  LUT5 #(
    .INIT(32'h45454544)) 
    \addr_reg[6]_i_1 
       (.I0(\addr_reg[6]_i_2_n_0 ),
        .I1(\FSM_onehot_state_reg[1]_i_2_n_0 ),
        .I2(\FSM_onehot_state_reg[6]_i_3_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[11] ),
        .O(\addr_reg[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT4 #(
    .INIT(16'hFFF6)) 
    \addr_reg[6]_i_2 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .O(\addr_reg[6]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT4 #(
    .INIT(16'h5554)) 
    \bit_count_reg[0]_i_1 
       (.I0(\u_i2c_master/bit_count_reg [0]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .O(\u_i2c_master/u_i2c_master/u_i2c_master/bit_count_next [0]));
  (* SOFT_HLUTNM = "soft_lutpair19" *) 
  LUT5 #(
    .INIT(32'hFE0000FE)) 
    \bit_count_reg[1]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .I3(\u_i2c_master/bit_count_reg [0]),
        .I4(\u_i2c_master/bit_count_reg [1]),
        .O(\u_i2c_master/u_i2c_master/u_i2c_master/bit_count_next [1]));
  LUT6 #(
    .INIT(64'hFEFEFE00000000FE)) 
    \bit_count_reg[2]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .I3(\u_i2c_master/bit_count_reg [1]),
        .I4(\u_i2c_master/bit_count_reg [0]),
        .I5(\u_i2c_master/bit_count_reg [2]),
        .O(\u_i2c_master/u_i2c_master/u_i2c_master/bit_count_next [2]));
  LUT5 #(
    .INIT(32'h54555555)) 
    \bit_count_reg[3]_i_1 
       (.I0(\addr_reg[6]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .I3(\bit_count_reg[3]_i_3_n_0 ),
        .I4(\FSM_onehot_state_reg[2]_i_2_n_0 ),
        .O(\bit_count_reg[3]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFF44444441)) 
    \bit_count_reg[3]_i_2 
       (.I0(\bit_count_reg[3]_i_4_n_0 ),
        .I1(\u_i2c_master/bit_count_reg [3]),
        .I2(\u_i2c_master/bit_count_reg [2]),
        .I3(\u_i2c_master/bit_count_reg [1]),
        .I4(\u_i2c_master/bit_count_reg [0]),
        .I5(\bit_count_reg[3]_i_5_n_0 ),
        .O(\u_i2c_master/u_i2c_master/u_i2c_master/bit_count_next [3]));
  LUT5 #(
    .INIT(32'h00001555)) 
    \bit_count_reg[3]_i_3 
       (.I0(\FSM_onehot_state_reg[4]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I2(\u_i2c_master/u_i2c_master/data_in_valid0 ),
        .I3(\u_i2c_master/u_i2c_master/data_in_ready0 ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\bit_count_reg[3]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair55" *) 
  LUT3 #(
    .INIT(8'h01)) 
    \bit_count_reg[3]_i_4 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\bit_count_reg[3]_i_4_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFD)) 
    \bit_count_reg[3]_i_5 
       (.I0(\bit_count_reg[3]_i_6_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[3] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[6] ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[1] ),
        .O(\bit_count_reg[3]_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair86" *) 
  LUT3 #(
    .INIT(8'h01)) 
    \bit_count_reg[3]_i_6 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[0] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[11] ),
        .O(\bit_count_reg[3]_i_6_n_0 ));
  LUT4 #(
    .INIT(16'hF740)) 
    bus_active_reg_i_1
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/sda_i_reg ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/scl_i_reg ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/last_sda_i_reg ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/bus_active_reg_reg_n_0 ),
        .O(bus_active_reg_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair86" *) 
  LUT4 #(
    .INIT(16'hAAAB)) 
    busy_reg_i_1
       (.I0(\addr_reg[6]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[11] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[0] ),
        .O(\u_i2c_master/u_i2c_master/u_i2c_master/busy_reg0 ));
  LUT6 #(
    .INIT(64'h0400000000000000)) 
    \cmd_mode_r[3]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/state [3]),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/state_reg_n_0_[2] ),
        .I3(\u_i2c_master/state_reg_n_0_[0] ),
        .I4(\u_i2c_master/u_i2c_master/state [1]),
        .I5(\u_i2c_master/u_i2c_master/state [0]),
        .O(\u_i2c_master/u_i2c_master/cmd_mode_r ));
  LUT6 #(
    .INIT(64'h00000000FFFF8A88)) 
    cmd_ready_reg_i_1
       (.I0(\FSM_onehot_state_reg[6]_i_3_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[11] ),
        .I2(\u_i2c_master/u_i2c_master/data_out_valid0 ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I4(\FSM_onehot_state_reg[0]_i_2_n_0 ),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/missed_ack_reg ),
        .O(cmd_ready_reg_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair10" *) 
  LUT5 #(
    .INIT(32'hFEFFFFFE)) 
    cmd_ready_reg_i_2
       (.I0(rst),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .O(\u_i2c_master/u_i2c_master/u_i2c_master/missed_ack_reg ));
  LUT1 #(
    .INIT(2'h1)) 
    \count[0]_i_1 
       (.I0(\u_tester/count_reg [0]),
        .O(p_0_in__1[0]));
  (* SOFT_HLUTNM = "soft_lutpair125" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \count[1]_i_1 
       (.I0(\u_tester/count_reg [0]),
        .I1(\u_tester/count_reg [1]),
        .O(p_0_in__1[1]));
  (* SOFT_HLUTNM = "soft_lutpair125" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \count[2]_i_1 
       (.I0(\u_tester/count_reg [2]),
        .I1(\u_tester/count_reg [1]),
        .I2(\u_tester/count_reg [0]),
        .O(p_0_in__1[2]));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \count[3]_i_1 
       (.I0(\u_tester/count_reg [3]),
        .I1(\u_tester/count_reg [0]),
        .I2(\u_tester/count_reg [1]),
        .I3(\u_tester/count_reg [2]),
        .O(p_0_in__1[3]));
  (* SOFT_HLUTNM = "soft_lutpair20" *) 
  LUT5 #(
    .INIT(32'h6AAAAAAA)) 
    \count[4]_i_1 
       (.I0(\u_tester/count_reg [4]),
        .I1(\u_tester/count_reg [2]),
        .I2(\u_tester/count_reg [1]),
        .I3(\u_tester/count_reg [0]),
        .I4(\u_tester/count_reg [3]),
        .O(p_0_in__1[4]));
  LUT6 #(
    .INIT(64'h6AAAAAAAAAAAAAAA)) 
    \count[5]_i_1 
       (.I0(\u_tester/count_reg [5]),
        .I1(\u_tester/count_reg [3]),
        .I2(\u_tester/count_reg [0]),
        .I3(\u_tester/count_reg [1]),
        .I4(\u_tester/count_reg [2]),
        .I5(\u_tester/count_reg [4]),
        .O(p_0_in__1[5]));
  (* SOFT_HLUTNM = "soft_lutpair173" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \count[6]_i_1 
       (.I0(\u_tester/count_reg [6]),
        .I1(\count[6]_i_2_n_0 ),
        .O(p_0_in__1[6]));
  LUT6 #(
    .INIT(64'h8000000000000000)) 
    \count[6]_i_2 
       (.I0(\u_tester/count_reg [5]),
        .I1(\u_tester/count_reg [3]),
        .I2(\u_tester/count_reg [0]),
        .I3(\u_tester/count_reg [1]),
        .I4(\u_tester/count_reg [2]),
        .I5(\u_tester/count_reg [4]),
        .O(\count[6]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000002)) 
    \ctrl0[15]_i_1 
       (.I0(\idovr[4]_i_2_n_0 ),
        .I1(addr[4]),
        .I2(addr[3]),
        .I3(addr[5]),
        .I4(addr[2]),
        .I5(addr[1]),
        .O(__do_out1127_out));
  (* SOFT_HLUTNM = "soft_lutpair140" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[0]_i_1 
       (.I0(\ctrl0_reg_n_0_[0] ),
        .I1(\gain_lna2_out[0] [0]),
        .I2(agc_en),
        .O(p_0_in[0]));
  (* SOFT_HLUTNM = "soft_lutpair122" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[10]_i_1 
       (.I0(\ctrl0_reg_n_0_[10] ),
        .I1(\gain_lna2_out[0] [10]),
        .I2(agc_en),
        .O(p_0_in[10]));
  (* SOFT_HLUTNM = "soft_lutpair120" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[11]_i_1 
       (.I0(\ctrl0_reg_n_0_[11] ),
        .I1(\gain_lna2_out[0] [11]),
        .I2(agc_en),
        .O(p_0_in[11]));
  (* SOFT_HLUTNM = "soft_lutpair119" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \ctrl0_rd[12]_i_1 
       (.I0(\gain_lna2_out[0] [12]),
        .I1(agc_en),
        .I2(\ctrl0_reg_n_0_[12] ),
        .O(p_0_in[12]));
  (* SOFT_HLUTNM = "soft_lutpair118" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[13]_i_1 
       (.I0(\ctrl0_reg_n_0_[13] ),
        .I1(\gain_lna2_out[0] [13]),
        .I2(agc_en),
        .O(p_0_in[13]));
  (* SOFT_HLUTNM = "soft_lutpair117" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[14]_i_1 
       (.I0(\ctrl0_reg_n_0_[14] ),
        .I1(\gain_lna2_out[0] [14]),
        .I2(agc_en),
        .O(p_0_in[14]));
  (* SOFT_HLUTNM = "soft_lutpair116" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[15]_i_1 
       (.I0(trx_auto),
        .I1(\gain_lna2_out[0] [15]),
        .I2(agc_en),
        .O(p_0_in[15]));
  (* SOFT_HLUTNM = "soft_lutpair139" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[1]_i_1 
       (.I0(fdd_en_b),
        .I1(\gain_lna2_out[0] [1]),
        .I2(agc_en),
        .O(p_0_in[1]));
  (* SOFT_HLUTNM = "soft_lutpair137" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[2]_i_1 
       (.I0(tx_hisel),
        .I1(\gain_lna2_out[0] [2]),
        .I2(agc_en),
        .O(p_0_in[2]));
  (* SOFT_HLUTNM = "soft_lutpair130" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[3]_i_1 
       (.I0(rx_hisel),
        .I1(\gain_lna2_out[0] [3]),
        .I2(agc_en),
        .O(p_0_in[3]));
  (* SOFT_HLUTNM = "soft_lutpair129" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[4]_i_1 
       (.I0(\ctrl0_reg_n_0_[4] ),
        .I1(\gain_lna2_out[0] [4]),
        .I2(agc_en),
        .O(p_0_in[4]));
  (* SOFT_HLUTNM = "soft_lutpair128" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[5]_i_1 
       (.I0(\ctrl0_reg_n_0_[5] ),
        .I1(\gain_lna2_out[0] [5]),
        .I2(agc_en),
        .O(p_0_in[5]));
  (* SOFT_HLUTNM = "soft_lutpair127" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[6]_i_1 
       (.I0(\ctrl0_reg_n_0_[6] ),
        .I1(\gain_lna2_out[0] [6]),
        .I2(agc_en),
        .O(p_0_in[6]));
  (* SOFT_HLUTNM = "soft_lutpair126" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[7]_i_1 
       (.I0(\ctrl0_reg_n_0_[7] ),
        .I1(\gain_lna2_out[0] [7]),
        .I2(agc_en),
        .O(p_0_in[7]));
  (* SOFT_HLUTNM = "soft_lutpair124" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[8]_i_1 
       (.I0(\ctrl0_reg_n_0_[8] ),
        .I1(\gain_lna2_out[0] [8]),
        .I2(agc_en),
        .O(p_0_in[8]));
  (* SOFT_HLUTNM = "soft_lutpair123" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl0_rd[9]_i_1 
       (.I0(\ctrl0_reg_n_0_[9] ),
        .I1(\gain_lna2_out[0] [9]),
        .I2(agc_en),
        .O(p_0_in[9]));
  FDRE \ctrl0_rd_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[0]),
        .Q(ctrl0_rd[0]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[10] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[10]),
        .Q(ctrl0_rd[10]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[11] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[11]),
        .Q(ctrl0_rd[11]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[12] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[12]),
        .Q(ctrl0_rd[12]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[13] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[13]),
        .Q(ctrl0_rd[13]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[14] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[14]),
        .Q(ctrl0_rd[14]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[15] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[15]),
        .Q(ctrl0_rd[15]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[1]),
        .Q(ctrl0_rd[1]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[2]),
        .Q(ctrl0_rd[2]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[3] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[3]),
        .Q(ctrl0_rd[3]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[4] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[4]),
        .Q(ctrl0_rd[4]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[5] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[5]),
        .Q(ctrl0_rd[5]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[6] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[6]),
        .Q(ctrl0_rd[6]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[7] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[7]),
        .Q(ctrl0_rd[7]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[8] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[8]),
        .Q(ctrl0_rd[8]),
        .R(1'b0));
  FDRE \ctrl0_rd_reg[9] 
       (.C(clk),
        .CE(1'b1),
        .D(p_0_in[9]),
        .Q(ctrl0_rd[9]),
        .R(1'b0));
  FDRE \ctrl0_reg[0] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[0]),
        .Q(\ctrl0_reg_n_0_[0] ),
        .R(rst));
  FDRE \ctrl0_reg[10] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[10]),
        .Q(\ctrl0_reg_n_0_[10] ),
        .R(rst));
  FDRE \ctrl0_reg[11] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[11]),
        .Q(\ctrl0_reg_n_0_[11] ),
        .R(rst));
  FDRE \ctrl0_reg[12] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[12]),
        .Q(\ctrl0_reg_n_0_[12] ),
        .R(rst));
  FDSE \ctrl0_reg[13] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[13]),
        .Q(\ctrl0_reg_n_0_[13] ),
        .S(rst));
  FDRE \ctrl0_reg[14] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[14]),
        .Q(\ctrl0_reg_n_0_[14] ),
        .R(rst));
  FDSE \ctrl0_reg[15] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[15]),
        .Q(trx_auto),
        .S(rst));
  FDRE \ctrl0_reg[1] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[1]),
        .Q(fdd_en_b),
        .R(rst));
  FDRE \ctrl0_reg[2] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[2]),
        .Q(tx_hisel),
        .R(rst));
  FDRE \ctrl0_reg[3] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[3]),
        .Q(rx_hisel),
        .R(rst));
  FDRE \ctrl0_reg[4] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[4]),
        .Q(\ctrl0_reg_n_0_[4] ),
        .R(rst));
  FDRE \ctrl0_reg[5] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[5]),
        .Q(\ctrl0_reg_n_0_[5] ),
        .R(rst));
  FDRE \ctrl0_reg[6] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[6]),
        .Q(\ctrl0_reg_n_0_[6] ),
        .R(rst));
  FDRE \ctrl0_reg[7] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[7]),
        .Q(\ctrl0_reg_n_0_[7] ),
        .R(rst));
  FDSE \ctrl0_reg[8] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[8]),
        .Q(\ctrl0_reg_n_0_[8] ),
        .S(rst));
  FDRE \ctrl0_reg[9] 
       (.C(clk),
        .CE(__do_out1127_out),
        .D(dati[9]),
        .Q(\ctrl0_reg_n_0_[9] ),
        .R(rst));
  LUT6 #(
    .INIT(64'h0000000000000002)) 
    \ctrl1[15]_i_1 
       (.I0(\test_ctrl[15]_i_2_n_0 ),
        .I1(addr[4]),
        .I2(addr[3]),
        .I3(addr[5]),
        .I4(addr[2]),
        .I5(addr[1]),
        .O(__do_out1123_out));
  (* SOFT_HLUTNM = "soft_lutpair116" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[0]_i_1 
       (.I0(\ctrl1_reg_n_0_[0] ),
        .I1(\gain_attn_out[0] [0]),
        .I2(agc_en),
        .O(\ctrl1_rd[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair129" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \ctrl1_rd[10]_i_1 
       (.I0(\gain_attn_out[0] [10]),
        .I1(agc_en),
        .I2(\ctrl1_reg_n_0_[10] ),
        .O(\ctrl1_rd[10]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair137" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \ctrl1_rd[11]_i_1 
       (.I0(\gain_attn_out[0] [11]),
        .I1(agc_en),
        .I2(\ctrl1_reg_n_0_[11] ),
        .O(\ctrl1_rd[11]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair138" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[12]_i_1 
       (.I0(\ctrl1_reg_n_0_[12] ),
        .I1(\gain_attn_out[0] [12]),
        .I2(agc_en),
        .O(\ctrl1_rd[12]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair139" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[13]_i_1 
       (.I0(p_0_in0_in),
        .I1(\gain_attn_out[0] [13]),
        .I2(agc_en),
        .O(\ctrl1_rd[13]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair140" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[14]_i_1 
       (.I0(\ctrl1_reg_n_0_[14] ),
        .I1(\gain_attn_out[0] [14]),
        .I2(agc_en),
        .O(\ctrl1_rd[14]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[15]_i_1 
       (.I0(\ctrl1_reg_n_0_[15] ),
        .I1(\gain_attn_out[0] [15]),
        .I2(agc_en),
        .O(\ctrl1_rd[15]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair117" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[1]_i_1 
       (.I0(\ctrl1_reg_n_0_[1] ),
        .I1(\gain_attn_out[0] [1]),
        .I2(agc_en),
        .O(\ctrl1_rd[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair122" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[2]_i_1 
       (.I0(\ctrl1_reg_n_0_[2] ),
        .I1(\gain_attn_out[0] [2]),
        .I2(agc_en),
        .O(\ctrl1_rd[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair119" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[3]_i_1 
       (.I0(\ctrl1_reg_n_0_[3] ),
        .I1(\gain_attn_out[0] [3]),
        .I2(agc_en),
        .O(\ctrl1_rd[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair120" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[4]_i_1 
       (.I0(\ctrl1_reg_n_0_[4] ),
        .I1(\gain_attn_out[0] [4]),
        .I2(agc_en),
        .O(\ctrl1_rd[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair126" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[5]_i_1 
       (.I0(\ctrl1_reg_n_0_[5] ),
        .I1(\gain_attn_out[0] [5]),
        .I2(agc_en),
        .O(\ctrl1_rd[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair123" *) 
  LUT3 #(
    .INIT(8'hF8)) 
    \ctrl1_rd[6]_i_1 
       (.I0(agc_en),
        .I1(\gain_attn_out[0] [6]),
        .I2(\ctrl1_reg_n_0_[6] ),
        .O(\ctrl1_rd[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair124" *) 
  LUT3 #(
    .INIT(8'hF8)) 
    \ctrl1_rd[7]_i_1 
       (.I0(agc_en),
        .I1(\gain_attn_out[0] [7]),
        .I2(\ctrl1_reg_n_0_[7] ),
        .O(\ctrl1_rd[7]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair127" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[8]_i_1 
       (.I0(\ctrl1_reg_n_0_[8] ),
        .I1(\gain_attn_out[0] [8]),
        .I2(agc_en),
        .O(\ctrl1_rd[8]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair128" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl1_rd[9]_i_1 
       (.I0(\ctrl1_reg_n_0_[9] ),
        .I1(\gain_attn_out[0] [9]),
        .I2(agc_en),
        .O(\ctrl1_rd[9]_i_1_n_0 ));
  FDRE \ctrl1_rd_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[0]_i_1_n_0 ),
        .Q(ctrl1_rd[0]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[10] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[10]_i_1_n_0 ),
        .Q(ctrl1_rd[10]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[11] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[11]_i_1_n_0 ),
        .Q(ctrl1_rd[11]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[12] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[12]_i_1_n_0 ),
        .Q(ctrl1_rd[12]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[13] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[13]_i_1_n_0 ),
        .Q(ctrl1_rd[13]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[14] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[14]_i_1_n_0 ),
        .Q(ctrl1_rd[14]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[15] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[15]_i_1_n_0 ),
        .Q(ctrl1_rd[15]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[1]_i_1_n_0 ),
        .Q(ctrl1_rd[1]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[2]_i_1_n_0 ),
        .Q(ctrl1_rd[2]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[3] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[3]_i_1_n_0 ),
        .Q(ctrl1_rd[3]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[4] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[4]_i_1_n_0 ),
        .Q(ctrl1_rd[4]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[5] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[5]_i_1_n_0 ),
        .Q(ctrl1_rd[5]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[6] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[6]_i_1_n_0 ),
        .Q(ctrl1_rd[6]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[7] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[7]_i_1_n_0 ),
        .Q(ctrl1_rd[7]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[8] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[8]_i_1_n_0 ),
        .Q(ctrl1_rd[8]),
        .R(1'b0));
  FDRE \ctrl1_rd_reg[9] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl1_rd[9]_i_1_n_0 ),
        .Q(ctrl1_rd[9]),
        .R(1'b0));
  FDRE \ctrl1_reg[0] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[0]),
        .Q(\ctrl1_reg_n_0_[0] ),
        .R(rst));
  FDRE \ctrl1_reg[10] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[10]),
        .Q(\ctrl1_reg_n_0_[10] ),
        .R(rst));
  FDRE \ctrl1_reg[11] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[11]),
        .Q(\ctrl1_reg_n_0_[11] ),
        .R(rst));
  FDRE \ctrl1_reg[12] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[12]),
        .Q(\ctrl1_reg_n_0_[12] ),
        .R(rst));
  FDRE \ctrl1_reg[13] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[13]),
        .Q(p_0_in0_in),
        .R(rst));
  FDRE \ctrl1_reg[14] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[14]),
        .Q(\ctrl1_reg_n_0_[14] ),
        .R(rst));
  FDRE \ctrl1_reg[15] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[15]),
        .Q(\ctrl1_reg_n_0_[15] ),
        .R(rst));
  FDRE \ctrl1_reg[1] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[1]),
        .Q(\ctrl1_reg_n_0_[1] ),
        .R(rst));
  FDRE \ctrl1_reg[2] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[2]),
        .Q(\ctrl1_reg_n_0_[2] ),
        .R(rst));
  FDRE \ctrl1_reg[3] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[3]),
        .Q(\ctrl1_reg_n_0_[3] ),
        .R(rst));
  FDRE \ctrl1_reg[4] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[4]),
        .Q(\ctrl1_reg_n_0_[4] ),
        .R(rst));
  FDRE \ctrl1_reg[5] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[5]),
        .Q(\ctrl1_reg_n_0_[5] ),
        .R(rst));
  FDRE \ctrl1_reg[6] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[6]),
        .Q(\ctrl1_reg_n_0_[6] ),
        .R(rst));
  FDRE \ctrl1_reg[7] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[7]),
        .Q(\ctrl1_reg_n_0_[7] ),
        .R(rst));
  FDRE \ctrl1_reg[8] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[8]),
        .Q(\ctrl1_reg_n_0_[8] ),
        .R(rst));
  FDRE \ctrl1_reg[9] 
       (.C(clk),
        .CE(__do_out1123_out),
        .D(dati[9]),
        .Q(\ctrl1_reg_n_0_[9] ),
        .R(rst));
  LUT6 #(
    .INIT(64'h0000000000000008)) 
    \ctrl2[15]_i_1 
       (.I0(\idovr[4]_i_2_n_0 ),
        .I1(addr[1]),
        .I2(addr[4]),
        .I3(addr[3]),
        .I4(addr[5]),
        .I5(addr[2]),
        .O(__do_out1118_out));
  (* SOFT_HLUTNM = "soft_lutpair130" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \ctrl2_rd[0]_i_1 
       (.I0(\gain_lna1_out[0] [0]),
        .I1(agc_en),
        .I2(\ctrl2_reg_n_0_[0] ),
        .O(\ctrl2_rd[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair118" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \ctrl2_rd[1]_i_1 
       (.I0(\gain_lna1_out[0] [1]),
        .I1(agc_en),
        .I2(\ctrl2_reg_n_0_[1] ),
        .O(\ctrl2_rd[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair114" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl2_rd[2]_i_1 
       (.I0(\ctrl2_reg_n_0_[2] ),
        .I1(\gain_lna1_out[0] [2]),
        .I2(agc_en),
        .O(\ctrl2_rd[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair115" *) 
  LUT3 #(
    .INIT(8'hEA)) 
    \ctrl2_rd[3]_i_1 
       (.I0(\ctrl2_reg_n_0_[3] ),
        .I1(\gain_lna1_out[0] [3]),
        .I2(agc_en),
        .O(\ctrl2_rd[3]_i_1_n_0 ));
  FDRE \ctrl2_rd_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_rd[0]_i_1_n_0 ),
        .Q(ctrl2_rd[0]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[10] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[10] ),
        .Q(ctrl2_rd[10]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[11] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[11] ),
        .Q(ctrl2_rd[11]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[12] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[12] ),
        .Q(ctrl2_rd[12]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[13] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[13] ),
        .Q(ctrl2_rd[13]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[14] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[14] ),
        .Q(ctrl2_rd[14]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[15] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[15] ),
        .Q(ctrl2_rd[15]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_rd[1]_i_1_n_0 ),
        .Q(ctrl2_rd[1]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_rd[2]_i_1_n_0 ),
        .Q(ctrl2_rd[2]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[3] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_rd[3]_i_1_n_0 ),
        .Q(ctrl2_rd[3]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[4] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[4] ),
        .Q(ctrl2_rd[4]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[5] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[5] ),
        .Q(ctrl2_rd[5]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[6] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[6] ),
        .Q(ctrl2_rd[6]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[7] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[7] ),
        .Q(ctrl2_rd[7]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[8] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[8] ),
        .Q(ctrl2_rd[8]),
        .R(1'b0));
  FDRE \ctrl2_rd_reg[9] 
       (.C(clk),
        .CE(1'b1),
        .D(\ctrl2_reg_n_0_[9] ),
        .Q(ctrl2_rd[9]),
        .R(1'b0));
  FDRE \ctrl2_reg[0] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[0]),
        .Q(\ctrl2_reg_n_0_[0] ),
        .R(rst));
  FDRE \ctrl2_reg[10] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[10]),
        .Q(\ctrl2_reg_n_0_[10] ),
        .R(rst));
  FDRE \ctrl2_reg[11] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[11]),
        .Q(\ctrl2_reg_n_0_[11] ),
        .R(rst));
  FDRE \ctrl2_reg[12] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[12]),
        .Q(\ctrl2_reg_n_0_[12] ),
        .R(rst));
  FDRE \ctrl2_reg[13] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[13]),
        .Q(\ctrl2_reg_n_0_[13] ),
        .R(rst));
  FDRE \ctrl2_reg[14] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[14]),
        .Q(\ctrl2_reg_n_0_[14] ),
        .R(rst));
  FDRE \ctrl2_reg[15] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[15]),
        .Q(\ctrl2_reg_n_0_[15] ),
        .R(rst));
  FDRE \ctrl2_reg[1] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[1]),
        .Q(\ctrl2_reg_n_0_[1] ),
        .R(rst));
  FDRE \ctrl2_reg[2] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[2]),
        .Q(\ctrl2_reg_n_0_[2] ),
        .R(rst));
  FDRE \ctrl2_reg[3] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[3]),
        .Q(\ctrl2_reg_n_0_[3] ),
        .R(rst));
  FDRE \ctrl2_reg[4] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[4]),
        .Q(\ctrl2_reg_n_0_[4] ),
        .R(rst));
  FDRE \ctrl2_reg[5] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[5]),
        .Q(\ctrl2_reg_n_0_[5] ),
        .R(rst));
  FDRE \ctrl2_reg[6] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[6]),
        .Q(\ctrl2_reg_n_0_[6] ),
        .R(rst));
  FDRE \ctrl2_reg[7] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[7]),
        .Q(\ctrl2_reg_n_0_[7] ),
        .R(rst));
  FDRE \ctrl2_reg[8] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[8]),
        .Q(\ctrl2_reg_n_0_[8] ),
        .R(rst));
  FDRE \ctrl2_reg[9] 
       (.C(clk),
        .CE(__do_out1118_out),
        .D(dati[9]),
        .Q(\ctrl2_reg_n_0_[9] ),
        .R(rst));
  LUT6 #(
    .INIT(64'h0000000000000008)) 
    \ctrl3[15]_i_1 
       (.I0(\test_ctrl[15]_i_2_n_0 ),
        .I1(addr[1]),
        .I2(addr[4]),
        .I3(addr[3]),
        .I4(addr[5]),
        .I5(addr[2]),
        .O(__do_out1113_out));
  FDRE \ctrl3_reg[0] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[0]),
        .Q(\ctrl3_reg_n_0_[0] ),
        .R(rst));
  FDRE \ctrl3_reg[10] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[10]),
        .Q(\ctrl3_reg_n_0_[10] ),
        .R(rst));
  FDRE \ctrl3_reg[11] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[11]),
        .Q(\ctrl3_reg_n_0_[11] ),
        .R(rst));
  FDRE \ctrl3_reg[12] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[12]),
        .Q(\ctrl3_reg_n_0_[12] ),
        .R(rst));
  FDRE \ctrl3_reg[13] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[13]),
        .Q(\ctrl3_reg_n_0_[13] ),
        .R(rst));
  FDRE \ctrl3_reg[14] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[14]),
        .Q(\ctrl3_reg_n_0_[14] ),
        .R(rst));
  FDRE \ctrl3_reg[15] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[15]),
        .Q(\ctrl3_reg_n_0_[15] ),
        .R(rst));
  FDRE \ctrl3_reg[1] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[1]),
        .Q(\ctrl3_reg_n_0_[1] ),
        .R(rst));
  FDRE \ctrl3_reg[2] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[2]),
        .Q(\ctrl3_reg_n_0_[2] ),
        .R(rst));
  FDRE \ctrl3_reg[3] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[3]),
        .Q(\ctrl3_reg_n_0_[3] ),
        .R(rst));
  FDRE \ctrl3_reg[4] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[4]),
        .Q(\ctrl3_reg_n_0_[4] ),
        .R(rst));
  FDRE \ctrl3_reg[5] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[5]),
        .Q(\ctrl3_reg_n_0_[5] ),
        .R(rst));
  FDRE \ctrl3_reg[6] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[6]),
        .Q(\ctrl3_reg_n_0_[6] ),
        .R(rst));
  FDRE \ctrl3_reg[7] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[7]),
        .Q(\ctrl3_reg_n_0_[7] ),
        .R(rst));
  FDRE \ctrl3_reg[8] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[8]),
        .Q(\ctrl3_reg_n_0_[8] ),
        .R(rst));
  FDRE \ctrl3_reg[9] 
       (.C(clk),
        .CE(__do_out1113_out),
        .D(dati[9]),
        .Q(\ctrl3_reg_n_0_[9] ),
        .R(rst));
  LUT6 #(
    .INIT(64'h0000066006600000)) 
    dat_cache1_carry__0_i_1
       (.I0(\w_outbus[22]_i_2_n_0 ),
        .I1(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[22] ),
        .I2(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[21] ),
        .I3(\w_outbus[21]_i_2_n_0 ),
        .I4(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[23] ),
        .I5(\w_outbus[23]_i_2_n_0 ),
        .O(dat_cache1_carry__0_i_1_n_0));
  LUT6 #(
    .INIT(64'h0000066006600000)) 
    dat_cache1_carry__0_i_2
       (.I0(\w_outbus[18]_i_2_n_0 ),
        .I1(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[18] ),
        .I2(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[19] ),
        .I3(\w_outbus[19]_i_2_n_0 ),
        .I4(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[20] ),
        .I5(\w_outbus[20]_i_2_n_0 ),
        .O(dat_cache1_carry__0_i_2_n_0));
  LUT6 #(
    .INIT(64'h0000066006600000)) 
    dat_cache1_carry__0_i_3
       (.I0(\w_outbus[15]_i_2_n_0 ),
        .I1(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[15] ),
        .I2(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[16] ),
        .I3(\w_outbus[16]_i_2_n_0 ),
        .I4(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[17] ),
        .I5(\w_outbus[17]_i_2_n_0 ),
        .O(dat_cache1_carry__0_i_3_n_0));
  LUT6 #(
    .INIT(64'h0000066006600000)) 
    dat_cache1_carry__0_i_4
       (.I0(\w_outbus[12]_i_2_n_0 ),
        .I1(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[12] ),
        .I2(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[13] ),
        .I3(\w_outbus[13]_i_2_n_0 ),
        .I4(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[14] ),
        .I5(\w_outbus[14]_i_2_n_0 ),
        .O(dat_cache1_carry__0_i_4_n_0));
  LUT6 #(
    .INIT(64'h0000900033330933)) 
    dat_cache1_carry__1_i_1
       (.I0(\ctrl2_reg_n_0_[15] ),
        .I1(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[31] ),
        .I2(\ctrl2_reg_n_0_[14] ),
        .I3(\w_outbus[31]_i_4_n_0 ),
        .I4(sync),
        .I5(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[30] ),
        .O(dat_cache1_carry__1_i_1_n_0));
  LUT6 #(
    .INIT(64'h0990000000000990)) 
    dat_cache1_carry__1_i_2
       (.I0(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[28] ),
        .I1(gpio_exp_out[28]),
        .I2(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[27] ),
        .I3(\w_outbus[27]_i_2_n_0 ),
        .I4(gpio_exp_out[29]),
        .I5(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[29] ),
        .O(dat_cache1_carry__1_i_2_n_0));
  LUT6 #(
    .INIT(64'h0000066006600000)) 
    dat_cache1_carry__1_i_3
       (.I0(\w_outbus[24]_i_2_n_0 ),
        .I1(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[24] ),
        .I2(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[25] ),
        .I3(\w_outbus[25]_i_2_n_0 ),
        .I4(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[26] ),
        .I5(\w_outbus[26]_i_2_n_0 ),
        .O(dat_cache1_carry__1_i_3_n_0));
  LUT6 #(
    .INIT(64'h0660000000000660)) 
    dat_cache1_carry_i_1
       (.I0(\w_outbus[10]_i_2_n_0 ),
        .I1(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[10] ),
        .I2(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[11] ),
        .I3(\w_outbus[11]_i_2_n_0 ),
        .I4(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[9] ),
        .I5(gpio_exp_out[9]),
        .O(dat_cache1_carry_i_1_n_0));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    dat_cache1_carry_i_2
       (.I0(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[7] ),
        .I1(gpio_exp_out[7]),
        .I2(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[8] ),
        .I3(gpio_exp_out[8]),
        .I4(gpio_exp_out[6]),
        .I5(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[6] ),
        .O(dat_cache1_carry_i_2_n_0));
  LUT6 #(
    .INIT(64'h6006000000006006)) 
    dat_cache1_carry_i_3
       (.I0(\w_outbus[3]_i_2_n_0 ),
        .I1(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[3] ),
        .I2(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[5] ),
        .I3(gpio_exp_out[5]),
        .I4(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[4] ),
        .I5(gpio_exp_out[4]),
        .O(dat_cache1_carry_i_3_n_0));
  LUT6 #(
    .INIT(64'h0000066006600000)) 
    dat_cache1_carry_i_4
       (.I0(\w_outbus[1]_i_2_n_0 ),
        .I1(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[1] ),
        .I2(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[2] ),
        .I3(\w_outbus[2]_i_2_n_0 ),
        .I4(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[0] ),
        .I5(dat_cache1_carry_i_5_n_0),
        .O(dat_cache1_carry_i_4_n_0));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    dat_cache1_carry_i_5
       (.I0(\dat_cache[21]_i_2_n_0 ),
        .I1(led_dnlink_on),
        .I2(\dat_cache[25]_i_2_n_0 ),
        .I3(\ctrl2_reg_n_0_[0] ),
        .O(dat_cache1_carry_i_5_n_0));
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[0]_i_1 
       (.I0(\ctrl2_reg_n_0_[0] ),
        .I1(\dat_cache[25]_i_2_n_0 ),
        .I2(led_dnlink_on),
        .I3(\dat_cache[21]_i_2_n_0 ),
        .O(gpio_exp_out[0]));
  (* SOFT_HLUTNM = "soft_lutpair69" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[10]_i_1 
       (.I0(ctrl0_rd[12]),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl2_reg_n_0_[10] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[10]));
  (* SOFT_HLUTNM = "soft_lutpair74" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[11]_i_1 
       (.I0(\ctrl1_reg_n_0_[9] ),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl2_reg_n_0_[11] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[11]));
  (* SOFT_HLUTNM = "soft_lutpair75" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[12]_i_1 
       (.I0(ctrl1_rd[10]),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl3_reg_n_0_[0] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[12]));
  (* SOFT_HLUTNM = "soft_lutpair76" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[13]_i_1 
       (.I0(ctrl1_rd[11]),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl3_reg_n_0_[1] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[13]));
  (* SOFT_HLUTNM = "soft_lutpair77" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[14]_i_1 
       (.I0(gain_attn2_local[0]),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl3_reg_n_0_[2] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[14]));
  (* SOFT_HLUTNM = "soft_lutpair84" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[15]_i_1 
       (.I0(gain_attn2_local[1]),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl3_reg_n_0_[3] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[15]));
  (* SOFT_HLUTNM = "soft_lutpair83" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[16]_i_1 
       (.I0(\ctrl1_reg_n_0_[0] ),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl3_reg_n_0_[4] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[16]));
  (* SOFT_HLUTNM = "soft_lutpair82" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[17]_i_1 
       (.I0(\ctrl1_reg_n_0_[2] ),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl3_reg_n_0_[5] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[17]));
  (* SOFT_HLUTNM = "soft_lutpair81" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[18]_i_1 
       (.I0(\ctrl1_reg_n_0_[3] ),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl3_reg_n_0_[6] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[18]));
  (* SOFT_HLUTNM = "soft_lutpair66" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[19]_i_1 
       (.I0(\ctrl1_reg_n_0_[4] ),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl3_reg_n_0_[7] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[19]));
  (* SOFT_HLUTNM = "soft_lutpair80" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[1]_i_1 
       (.I0(\ctrl2_reg_n_0_[1] ),
        .I1(\dat_cache[25]_i_2_n_0 ),
        .I2(led_uplink_on),
        .I3(\dat_cache[21]_i_2_n_0 ),
        .O(gpio_exp_out[1]));
  (* SOFT_HLUTNM = "soft_lutpair79" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[20]_i_1 
       (.I0(rx_hisel),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl3_reg_n_0_[8] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[20]));
  (* SOFT_HLUTNM = "soft_lutpair78" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[21]_i_1 
       (.I0(tx_hisel),
        .I1(\dat_cache[21]_i_2_n_0 ),
        .I2(\ctrl3_reg_n_0_[9] ),
        .I3(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[21]));
  LUT5 #(
    .INIT(32'hFFFFFFDF)) 
    \dat_cache[21]_i_2 
       (.I0(dut_pgood),
        .I1(rfmod_id[0]),
        .I2(rfmod_id[1]),
        .I3(rfmod_id[2]),
        .I4(rfmod_id[3]),
        .O(\dat_cache[21]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair42" *) 
  LUT5 #(
    .INIT(32'h8080FF80)) 
    \dat_cache[22]_i_1 
       (.I0(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .I1(ctrl2_rd[0]),
        .I2(dut_pgood),
        .I3(led_dnlink_on),
        .I4(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[22]));
  (* SOFT_HLUTNM = "soft_lutpair43" *) 
  LUT5 #(
    .INIT(32'h8080FF80)) 
    \dat_cache[23]_i_1 
       (.I0(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .I1(ctrl2_rd[1]),
        .I2(dut_pgood),
        .I3(led_uplink_on),
        .I4(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[23]));
  (* SOFT_HLUTNM = "soft_lutpair44" *) 
  LUT5 #(
    .INIT(32'h8080FF80)) 
    \dat_cache[24]_i_1 
       (.I0(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .I1(\ctrl2_reg_n_0_[2] ),
        .I2(dut_pgood),
        .I3(led_error),
        .I4(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[24]));
  (* SOFT_HLUTNM = "soft_lutpair45" *) 
  LUT5 #(
    .INIT(32'h8080FF80)) 
    \dat_cache[25]_i_1 
       (.I0(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .I1(\ctrl2_reg_n_0_[3] ),
        .I2(dut_pgood),
        .I3(led_good),
        .I4(\dat_cache[25]_i_2_n_0 ),
        .O(gpio_exp_out[25]));
  LUT5 #(
    .INIT(32'hFFFFFFDF)) 
    \dat_cache[25]_i_2 
       (.I0(dut_pgood),
        .I1(rfmod_id[1]),
        .I2(rfmod_id[0]),
        .I3(rfmod_id[2]),
        .I4(rfmod_id[3]),
        .O(\dat_cache[25]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair46" *) 
  LUT5 #(
    .INIT(32'hAA808080)) 
    \dat_cache[26]_i_1 
       (.I0(dut_pgood),
        .I1(\rfmod_oe[8]_INST_0_i_1_n_0 ),
        .I2(\ctrl3_reg_n_0_[10] ),
        .I3(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .I4(\ctrl2_reg_n_0_[4] ),
        .O(gpio_exp_out[26]));
  (* SOFT_HLUTNM = "soft_lutpair47" *) 
  LUT5 #(
    .INIT(32'hAA808080)) 
    \dat_cache[27]_i_1 
       (.I0(dut_pgood),
        .I1(\rfmod_oe[8]_INST_0_i_1_n_0 ),
        .I2(\ctrl3_reg_n_0_[11] ),
        .I3(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .I4(\ctrl2_reg_n_0_[5] ),
        .O(gpio_exp_out[27]));
  (* SOFT_HLUTNM = "soft_lutpair63" *) 
  LUT4 #(
    .INIT(16'h4440)) 
    \dat_cache[28]_i_1 
       (.I0(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I1(dut_pgood),
        .I2(sync),
        .I3(\ctrl2_reg_n_0_[12] ),
        .O(gpio_exp_out[28]));
  (* SOFT_HLUTNM = "soft_lutpair64" *) 
  LUT4 #(
    .INIT(16'h4440)) 
    \dat_cache[29]_i_1 
       (.I0(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I1(dut_pgood),
        .I2(\ctrl2_reg_n_0_[13] ),
        .I3(sync),
        .O(gpio_exp_out[29]));
  (* SOFT_HLUTNM = "soft_lutpair65" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[2]_i_1 
       (.I0(\ctrl2_reg_n_0_[2] ),
        .I1(\dat_cache[25]_i_2_n_0 ),
        .I2(led_error),
        .I3(\dat_cache[21]_i_2_n_0 ),
        .O(gpio_exp_out[2]));
  (* SOFT_HLUTNM = "soft_lutpair63" *) 
  LUT4 #(
    .INIT(16'h0008)) 
    \dat_cache[30]_i_1 
       (.I0(\ctrl2_reg_n_0_[14] ),
        .I1(dut_pgood),
        .I2(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I3(sync),
        .O(gpio_exp_out[30]));
  LUT3 #(
    .INIT(8'h54)) 
    \dat_cache[31]_i_1 
       (.I0(gpio_exp_busy),
        .I1(\u_spi_gpio_exp_master/dat_cache1_carry__1_n_1 ),
        .I2(gpio_exp_trigger),
        .O(\u_spi_gpio_exp_master/dat_cache ));
  (* SOFT_HLUTNM = "soft_lutpair64" *) 
  LUT4 #(
    .INIT(16'h0008)) 
    \dat_cache[31]_i_2 
       (.I0(\ctrl2_reg_n_0_[15] ),
        .I1(dut_pgood),
        .I2(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I3(sync),
        .O(gpio_exp_out[31]));
  (* SOFT_HLUTNM = "soft_lutpair71" *) 
  LUT4 #(
    .INIT(16'h22F2)) 
    \dat_cache[3]_i_1 
       (.I0(\ctrl2_reg_n_0_[3] ),
        .I1(\dat_cache[25]_i_2_n_0 ),
        .I2(led_good),
        .I3(\dat_cache[21]_i_2_n_0 ),
        .O(gpio_exp_out[3]));
  LUT6 #(
    .INIT(64'h0000020000000000)) 
    \dat_cache[4]_i_1 
       (.I0(\ctrl2_reg_n_0_[4] ),
        .I1(rfmod_id[3]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[0]),
        .I4(rfmod_id[1]),
        .I5(dut_pgood),
        .O(gpio_exp_out[4]));
  LUT6 #(
    .INIT(64'h0000020000000000)) 
    \dat_cache[5]_i_1 
       (.I0(\ctrl2_reg_n_0_[5] ),
        .I1(rfmod_id[3]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[0]),
        .I4(rfmod_id[1]),
        .I5(dut_pgood),
        .O(gpio_exp_out[5]));
  LUT6 #(
    .INIT(64'h0000020000000000)) 
    \dat_cache[6]_i_1 
       (.I0(\ctrl2_reg_n_0_[6] ),
        .I1(rfmod_id[3]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[0]),
        .I4(rfmod_id[1]),
        .I5(dut_pgood),
        .O(gpio_exp_out[6]));
  LUT6 #(
    .INIT(64'h0000020000000000)) 
    \dat_cache[7]_i_1 
       (.I0(\ctrl2_reg_n_0_[7] ),
        .I1(rfmod_id[3]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[0]),
        .I4(rfmod_id[1]),
        .I5(dut_pgood),
        .O(gpio_exp_out[7]));
  LUT6 #(
    .INIT(64'h0000020000000000)) 
    \dat_cache[8]_i_1 
       (.I0(\ctrl2_reg_n_0_[8] ),
        .I1(rfmod_id[3]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[0]),
        .I4(rfmod_id[1]),
        .I5(dut_pgood),
        .O(gpio_exp_out[8]));
  LUT6 #(
    .INIT(64'h0000020000000000)) 
    \dat_cache[9]_i_1 
       (.I0(\ctrl2_reg_n_0_[9] ),
        .I1(rfmod_id[3]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[0]),
        .I4(rfmod_id[1]),
        .I5(dut_pgood),
        .O(gpio_exp_out[9]));
  LUT6 #(
    .INIT(64'h0000000000000082)) 
    data_in_ready_reg_i_1
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/data_in_ready_next ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I5(rst),
        .O(data_in_ready_reg_i_1_n_0));
  LUT5 #(
    .INIT(32'h20000000)) 
    \data_o0[7]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/state [0]),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [1]),
        .I4(\u_i2c_master/u_i2c_master/data_out_valid ),
        .O(\data_o0[7]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hBA)) 
    \data_o0[7]_i_1__0 
       (.I0(rst),
        .I1(i2c_busy),
        .I2(i2c_cmd_en),
        .O(\data_o0[7]_i_1__0_n_0 ));
  LUT4 #(
    .INIT(16'h0008)) 
    \data_o0[7]_i_2 
       (.I0(\state[1]_i_2_n_0 ),
        .I1(\u_i2c_master/state_reg_n_0_[2] ),
        .I2(\u_i2c_master/state_reg_n_0_[0] ),
        .I3(\u_i2c_master/state_reg_n_0_[1] ),
        .O(\u_i2c_master/ext_cmd_queued0 ));
  LUT6 #(
    .INIT(64'h0000000000800000)) 
    \data_o1[7]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/data_out_valid ),
        .I1(\u_i2c_master/u_i2c_master/state [1]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/u_i2c_master/data_out_last ),
        .I5(\u_i2c_master/u_i2c_master/state [0]),
        .O(\data_o1[7]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hB8)) 
    data_out_last_reg_i_1
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg ),
        .I1(\data_out_reg[7]_i_1_n_0 ),
        .I2(\u_i2c_master/data_out_last_reg ),
        .O(data_out_last_reg_i_1_n_0));
  LUT6 #(
    .INIT(64'h0000000000000004)) 
    \data_out_reg[7]_i_1 
       (.I0(\addr_reg[6]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .I2(\u_i2c_master/bit_count_reg [3]),
        .I3(\u_i2c_master/bit_count_reg [2]),
        .I4(\u_i2c_master/bit_count_reg [1]),
        .I5(\u_i2c_master/bit_count_reg [0]),
        .O(\data_out_reg[7]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFEAAAAAAAAAAAA)) 
    data_out_valid_reg_i_1
       (.I0(\data_out_reg[7]_i_1_n_0 ),
        .I1(\addr_reg[6]_i_2_n_0 ),
        .I2(data_out_valid_reg_i_2_n_0),
        .I3(data_out_valid_reg_i_3_n_0),
        .I4(\u_i2c_master/u_i2c_master/data_out_valid0 ),
        .I5(\wr_addr_reg[1]_i_2_n_0 ),
        .O(data_out_valid_reg_i_1_n_0));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    data_out_valid_reg_i_2
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[11] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[0] ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[6] ),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[7] ),
        .O(data_out_valid_reg_i_2_n_0));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFFFFFE)) 
    data_out_valid_reg_i_3
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[1] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[3] ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[10] ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .O(data_out_valid_reg_i_3_n_0));
  LUT4 #(
    .INIT(16'hF888)) 
    \data_reg[0]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[0] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [0]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\data_reg[0]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \data_reg[1]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[1] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [1]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\data_reg[1]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \data_reg[2]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[2] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [2]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\data_reg[2]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \data_reg[3]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[3] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [3]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\data_reg[3]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \data_reg[4]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[4] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [4]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\data_reg[4]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \data_reg[5]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[5] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [5]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\data_reg[5]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \data_reg[6]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[6] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [6]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\data_reg[6]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h00001001)) 
    \data_reg[7]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I4(\bit_count_reg[3]_i_3_n_0 ),
        .O(\data_reg[7]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'hF888)) 
    \data_reg[7]_i_2 
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[7] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [7]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .O(\data_reg[7]_i_2_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__0_i_1
       (.I0(\u_i2c_master/delay_reg [8]),
        .O(delay_next0_carry__0_i_1_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__0_i_2
       (.I0(\u_i2c_master/delay_reg [7]),
        .O(delay_next0_carry__0_i_2_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__0_i_3
       (.I0(\u_i2c_master/delay_reg [6]),
        .O(delay_next0_carry__0_i_3_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__0_i_4
       (.I0(\u_i2c_master/delay_reg [5]),
        .O(delay_next0_carry__0_i_4_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__1_i_1
       (.I0(\u_i2c_master/delay_reg [12]),
        .O(delay_next0_carry__1_i_1_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__1_i_2
       (.I0(\u_i2c_master/delay_reg [11]),
        .O(delay_next0_carry__1_i_2_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__1_i_3
       (.I0(\u_i2c_master/delay_reg [10]),
        .O(delay_next0_carry__1_i_3_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__1_i_4
       (.I0(\u_i2c_master/delay_reg [9]),
        .O(delay_next0_carry__1_i_4_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__2_i_1
       (.I0(\u_i2c_master/delay_reg [16]),
        .O(delay_next0_carry__2_i_1_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__2_i_2
       (.I0(\u_i2c_master/delay_reg [15]),
        .O(delay_next0_carry__2_i_2_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__2_i_3
       (.I0(\u_i2c_master/delay_reg [14]),
        .O(delay_next0_carry__2_i_3_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry__2_i_4
       (.I0(\u_i2c_master/delay_reg [13]),
        .O(delay_next0_carry__2_i_4_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry_i_1
       (.I0(\u_i2c_master/delay_reg [4]),
        .O(delay_next0_carry_i_1_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry_i_2
       (.I0(\u_i2c_master/delay_reg [3]),
        .O(delay_next0_carry_i_2_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry_i_3
       (.I0(\u_i2c_master/delay_reg [2]),
        .O(delay_next0_carry_i_3_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    delay_next0_carry_i_4
       (.I0(\u_i2c_master/delay_reg [1]),
        .O(delay_next0_carry_i_4_n_0));
  (* SOFT_HLUTNM = "soft_lutpair143" *) 
  LUT2 #(
    .INIT(4'h1)) 
    \delay_reg[0]_i_1 
       (.I0(\u_i2c_master/delay_reg [0]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair176" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \delay_reg[10]_i_1 
       (.I0(\u_i2c_master/delay_next0 [10]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[10]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair176" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \delay_reg[11]_i_1 
       (.I0(\u_i2c_master/delay_next0 [11]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[11]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair183" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \delay_reg[12]_i_1 
       (.I0(\u_i2c_master/delay_next0 [12]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[12]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair183" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \delay_reg[13]_i_1 
       (.I0(\u_i2c_master/delay_next0 [13]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[13]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair185" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \delay_reg[14]_i_1 
       (.I0(\u_i2c_master/delay_next0 [14]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[14]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair185" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \delay_reg[15]_i_1 
       (.I0(\u_i2c_master/delay_next0 [15]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[15]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h5544555554544444)) 
    \delay_reg[16]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/delay_scl_reg ),
        .I1(\delay_reg[16]_i_3_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I3(\delay_reg[16]_i_4_n_0 ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .O(\delay_reg[16]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h2)) 
    \delay_reg[16]_i_2 
       (.I0(\u_i2c_master/delay_next0 [16]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[16]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h55885589FFFFFFFF)) 
    \delay_reg[16]_i_3 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I4(\FSM_onehot_state_reg[2]_i_2_n_0 ),
        .I5(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[16]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair101" *) 
  LUT3 #(
    .INIT(8'hFE)) 
    \delay_reg[16]_i_4 
       (.I0(\FSM_sequential_phy_state_reg[3]_i_4_n_0 ),
        .I1(\FSM_sequential_phy_state_reg[0]_i_2_n_0 ),
        .I2(\FSM_sequential_phy_state_reg[0]_i_3_n_0 ),
        .O(\delay_reg[16]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hEFFFFFFFEFFF0000)) 
    \delay_reg[1]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I4(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .I5(\u_i2c_master/delay_next0 [1]),
        .O(\delay_reg[1]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h10FF1000)) 
    \delay_reg[2]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I3(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .I4(\u_i2c_master/delay_next0 [2]),
        .O(\delay_reg[2]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hEFFFFFFFEFFF0000)) 
    \delay_reg[3]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I4(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .I5(\u_i2c_master/delay_next0 [3]),
        .O(\delay_reg[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair166" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \delay_reg[4]_i_1 
       (.I0(\u_i2c_master/delay_next0 [4]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair170" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \delay_reg[5]_i_1 
       (.I0(\u_i2c_master/delay_next0 [5]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair170" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \delay_reg[6]_i_1 
       (.I0(\u_i2c_master/delay_next0 [6]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair143" *) 
  LUT2 #(
    .INIT(4'hE)) 
    \delay_reg[7]_i_1 
       (.I0(\u_i2c_master/delay_next0 [7]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[7]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT5 #(
    .INIT(32'h10FF1000)) 
    \delay_reg[8]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I3(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .I4(\u_i2c_master/delay_next0 [8]),
        .O(\delay_reg[8]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair166" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \delay_reg[9]_i_1 
       (.I0(\u_i2c_master/delay_next0 [9]),
        .I1(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(\delay_reg[9]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h404F4040)) 
    delay_scl_reg_i_1
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/scl_i_reg ),
        .I1(scl_t),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/delay_scl_reg ),
        .I3(delay_scl_reg_i_2_n_0),
        .I4(\FSM_sequential_phy_state_reg[3]_i_3_n_0 ),
        .O(delay_scl_reg_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair70" *) 
  LUT4 #(
    .INIT(16'hDADF)) 
    delay_scl_reg_i_2
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .O(delay_scl_reg_i_2_n_0));
  LUT4 #(
    .INIT(16'h8A88)) 
    dut_tdo_INST_0
       (.I0(rfmod_in[9]),
        .I1(prog_jen),
        .I2(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I3(dut_ten),
        .O(dut_tdo));
  LUT4 #(
    .INIT(16'h7774)) 
    en_i_1
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[5] ),
        .I1(gpio_exp_busy),
        .I2(\u_spi_gpio_exp_master/dat_cache1_carry__1_n_1 ),
        .I3(gpio_exp_trigger),
        .O(en_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT5 #(
    .INIT(32'hBABFEFEA)) 
    \err[0]_i_1 
       (.I0(p_29_in[8]),
        .I1(rfmod_in[1]),
        .I2(test_dir),
        .I3(rfmod_in[5]),
        .I4(\u_tester/chk_out [0]),
        .O(\err[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT5 #(
    .INIT(32'hBABFEFEA)) 
    \err[1]_i_1 
       (.I0(p_29_in[9]),
        .I1(rfmod_in[2]),
        .I2(test_dir),
        .I3(rfmod_in[6]),
        .I4(\u_tester/chk_out [1]),
        .O(\err[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT5 #(
    .INIT(32'hBABFEFEA)) 
    \err[2]_i_1 
       (.I0(p_29_in[10]),
        .I1(rfmod_in[3]),
        .I2(test_dir),
        .I3(rfmod_in[7]),
        .I4(\u_tester/chk_out [2]),
        .O(\err[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT5 #(
    .INIT(32'hBABFEFEA)) 
    \err[3]_i_1 
       (.I0(p_29_in[11]),
        .I1(rfmod_in[4]),
        .I2(test_dir),
        .I3(rfmod_in[8]),
        .I4(\u_tester/chk_out [3]),
        .O(\err[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT5 #(
    .INIT(32'hBABFEFEA)) 
    \err[4]_i_1 
       (.I0(p_29_in[12]),
        .I1(rfmod_in[9]),
        .I2(test_dir),
        .I3(rfmod_in[13]),
        .I4(\u_tester/chk_out [4]),
        .O(\err[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT5 #(
    .INIT(32'hBABFEFEA)) 
    \err[5]_i_1 
       (.I0(p_29_in[13]),
        .I1(rfmod_in[10]),
        .I2(test_dir),
        .I3(rfmod_in[14]),
        .I4(\u_tester/chk_out [5]),
        .O(\err[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT5 #(
    .INIT(32'hBABFEFEA)) 
    \err[6]_i_1 
       (.I0(p_29_in[14]),
        .I1(rfmod_in[11]),
        .I2(test_dir),
        .I3(rfmod_in[15]),
        .I4(\u_tester/chk_out [6]),
        .O(\err[6]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'hEEEF)) 
    \err[7]_i_1 
       (.I0(rst),
        .I1(test_stat[22]),
        .I2(\u_tester/st_cur [1]),
        .I3(\u_tester/st_cur [0]),
        .O(\err[7]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h8)) 
    \err[7]_i_2 
       (.I0(\u_tester/st_cur [1]),
        .I1(\u_tester/st_cur [0]),
        .O(test_stat[27]));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT5 #(
    .INIT(32'hBABFEFEA)) 
    \err[7]_i_3 
       (.I0(p_29_in[15]),
        .I1(rfmod_in[12]),
        .I2(test_dir),
        .I3(rfmod_in[16]),
        .I4(\u_tester/chk_out [7]),
        .O(\err[7]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h00000000E2E200E2)) 
    error__0_i_1
       (.I0(i2c_error),
        .I1(\u_i2c_master/ext_cmd_queued0 ),
        .I2(\u_i2c_master/u_i2c_master/error_reg_n_0 ),
        .I3(i2c_cmd_en),
        .I4(i2c_busy),
        .I5(rst),
        .O(error__0_i_1_n_0));
  LUT4 #(
    .INIT(16'h0054)) 
    error_i_1
       (.I0(rst),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/missed_ack_reg_reg_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/error_reg_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/cmd_mode_r ),
        .O(error_i_1_n_0));
  LUT3 #(
    .INIT(8'h74)) 
    ext_cmd_queued_i_1
       (.I0(\u_i2c_master/ext_cmd_queued0 ),
        .I1(i2c_busy),
        .I2(i2c_cmd_en),
        .O(ext_cmd_queued_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair115" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \gain_attn2_local[0]_i_1 
       (.I0(\gain_attn_out[1] [10]),
        .I1(agc_en),
        .I2(\ctrl1_reg_n_0_[6] ),
        .O(\gain_attn2_local[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair114" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \gain_attn2_local[1]_i_1 
       (.I0(\gain_attn_out[1] [11]),
        .I1(agc_en),
        .I2(\ctrl1_reg_n_0_[7] ),
        .O(\gain_attn2_local[1]_i_1_n_0 ));
  FDRE \gain_attn2_local_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\gain_attn2_local[0]_i_1_n_0 ),
        .Q(gain_attn2_local[0]),
        .R(1'b0));
  FDRE \gain_attn2_local_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\gain_attn2_local[1]_i_1_n_0 ),
        .Q(gain_attn2_local[1]),
        .R(1'b0));
  LUT6 #(
    .INIT(64'h0000000000000800)) 
    \gpio_dato[16]_i_1 
       (.I0(\test_ctrl[15]_i_2_n_0 ),
        .I1(addr[3]),
        .I2(addr[5]),
        .I3(addr[2]),
        .I4(addr[4]),
        .I5(addr[1]),
        .O(__do_out188_out));
  FDRE \gpio_dato_reg[10] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[9]),
        .Q(\gpio_dato_reg_n_0_[10] ),
        .R(rst));
  FDRE \gpio_dato_reg[11] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[10]),
        .Q(\gpio_dato_reg_n_0_[11] ),
        .R(rst));
  FDRE \gpio_dato_reg[12] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[11]),
        .Q(\gpio_dato_reg_n_0_[12] ),
        .R(rst));
  FDRE \gpio_dato_reg[13] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[12]),
        .Q(\gpio_dato_reg_n_0_[13] ),
        .R(rst));
  FDRE \gpio_dato_reg[14] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[13]),
        .Q(\gpio_dato_reg_n_0_[14] ),
        .R(rst));
  FDRE \gpio_dato_reg[15] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[14]),
        .Q(\gpio_dato_reg_n_0_[15] ),
        .R(rst));
  FDRE \gpio_dato_reg[16] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[15]),
        .Q(\gpio_dato_reg_n_0_[16] ),
        .R(rst));
  FDRE \gpio_dato_reg[1] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[0]),
        .Q(\gpio_dato_reg_n_0_[1] ),
        .R(rst));
  FDRE \gpio_dato_reg[2] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[1]),
        .Q(\gpio_dato_reg_n_0_[2] ),
        .R(rst));
  FDRE \gpio_dato_reg[3] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[2]),
        .Q(\gpio_dato_reg_n_0_[3] ),
        .R(rst));
  FDRE \gpio_dato_reg[4] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[3]),
        .Q(\gpio_dato_reg_n_0_[4] ),
        .R(rst));
  FDRE \gpio_dato_reg[5] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[4]),
        .Q(\gpio_dato_reg_n_0_[5] ),
        .R(rst));
  FDRE \gpio_dato_reg[6] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[5]),
        .Q(data0),
        .R(rst));
  FDRE \gpio_dato_reg[7] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[6]),
        .Q(\gpio_dato_reg_n_0_[7] ),
        .R(rst));
  FDRE \gpio_dato_reg[8] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[7]),
        .Q(\gpio_dato_reg_n_0_[8] ),
        .R(rst));
  FDRE \gpio_dato_reg[9] 
       (.C(clk),
        .CE(__do_out188_out),
        .D(dati[8]),
        .Q(\gpio_dato_reg_n_0_[9] ),
        .R(rst));
  LUT6 #(
    .INIT(64'h0000000000800000)) 
    \gpio_dir[16]_i_1 
       (.I0(\idovr[4]_i_2_n_0 ),
        .I1(addr[1]),
        .I2(addr[3]),
        .I3(addr[5]),
        .I4(addr[2]),
        .I5(addr[4]),
        .O(__do_out183_out));
  FDRE \gpio_dir_reg[10] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[9]),
        .Q(\gpio_dir_reg_n_0_[10] ),
        .R(rst));
  FDRE \gpio_dir_reg[11] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[10]),
        .Q(\gpio_dir_reg_n_0_[11] ),
        .R(rst));
  FDRE \gpio_dir_reg[12] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[11]),
        .Q(\gpio_dir_reg_n_0_[12] ),
        .R(rst));
  FDRE \gpio_dir_reg[13] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[12]),
        .Q(\gpio_dir_reg_n_0_[13] ),
        .R(rst));
  FDRE \gpio_dir_reg[14] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[13]),
        .Q(\gpio_dir_reg_n_0_[14] ),
        .R(rst));
  FDRE \gpio_dir_reg[15] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[14]),
        .Q(\gpio_dir_reg_n_0_[15] ),
        .R(rst));
  FDRE \gpio_dir_reg[16] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[15]),
        .Q(\gpio_dir_reg_n_0_[16] ),
        .R(rst));
  FDRE \gpio_dir_reg[1] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[0]),
        .Q(\gpio_dir_reg_n_0_[1] ),
        .R(rst));
  FDRE \gpio_dir_reg[2] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[1]),
        .Q(p_1_in),
        .R(rst));
  FDRE \gpio_dir_reg[3] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[2]),
        .Q(\gpio_dir_reg_n_0_[3] ),
        .R(rst));
  FDRE \gpio_dir_reg[4] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[3]),
        .Q(\gpio_dir_reg_n_0_[4] ),
        .R(rst));
  FDRE \gpio_dir_reg[5] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[4]),
        .Q(p_2_in),
        .R(rst));
  FDRE \gpio_dir_reg[6] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[5]),
        .Q(\gpio_dir_reg_n_0_[6] ),
        .R(rst));
  FDRE \gpio_dir_reg[7] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[6]),
        .Q(\gpio_dir_reg_n_0_[7] ),
        .R(rst));
  FDRE \gpio_dir_reg[8] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[7]),
        .Q(\gpio_dir_reg_n_0_[8] ),
        .R(rst));
  FDRE \gpio_dir_reg[9] 
       (.C(clk),
        .CE(__do_out183_out),
        .D(dati[8]),
        .Q(\gpio_dir_reg_n_0_[9] ),
        .R(rst));
  LUT5 #(
    .INIT(32'h00800000)) 
    gpio_exp_trigger_i_1
       (.I0(gpio_exp_trigger_i_2_n_0),
        .I1(en),
        .I2(wr),
        .I3(addr[0]),
        .I4(rdy),
        .O(gpio_exp_trigger0));
  (* SOFT_HLUTNM = "soft_lutpair35" *) 
  LUT5 #(
    .INIT(32'h00000008)) 
    gpio_exp_trigger_i_2
       (.I0(addr[1]),
        .I1(addr[4]),
        .I2(addr[2]),
        .I3(addr[5]),
        .I4(addr[3]),
        .O(gpio_exp_trigger_i_2_n_0));
  FDRE gpio_exp_trigger_reg
       (.C(clk),
        .CE(1'b1),
        .D(gpio_exp_trigger0),
        .Q(gpio_exp_trigger),
        .R(1'b0));
  (* SOFT_HLUTNM = "soft_lutpair136" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_addr_s[0]_i_1 
       (.I0(i2c_addr_s[0]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_6_11_n_3 ),
        .O(\i2c_addr_s[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair135" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_addr_s[1]_i_1 
       (.I0(i2c_addr_s[1]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_6_11_n_2 ),
        .O(\i2c_addr_s[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair135" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_addr_s[2]_i_1 
       (.I0(i2c_addr_s[2]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_6_11_n_5 ),
        .O(\i2c_addr_s[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair134" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_addr_s[3]_i_1 
       (.I0(i2c_addr_s[3]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_6_11_n_4 ),
        .O(\i2c_addr_s[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair133" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_addr_s[4]_i_1 
       (.I0(i2c_addr_s[4]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_12_15_n_1 ),
        .O(\i2c_addr_s[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair132" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_addr_s[5]_i_1 
       (.I0(i2c_addr_s[5]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_12_15_n_0 ),
        .O(\i2c_addr_s[5]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h0000000E)) 
    \i2c_addr_s[6]_i_1 
       (.I0(\i2c_addr_s[6]_i_3_n_0 ),
        .I1(i2c_busy),
        .I2(\u_i2c_master/state_reg_n_0_[0] ),
        .I3(\u_i2c_master/state_reg_n_0_[2] ),
        .I4(\u_i2c_master/state_reg_n_0_[1] ),
        .O(\i2c_addr_s[6]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000000400)) 
    \i2c_addr_s[6]_i_1__0 
       (.I0(addr[1]),
        .I1(\test_ctrl[15]_i_2_n_0 ),
        .I2(addr[4]),
        .I3(addr[3]),
        .I4(addr[5]),
        .I5(addr[2]),
        .O(i2c_addr_s0));
  (* SOFT_HLUTNM = "soft_lutpair131" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_addr_s[6]_i_2 
       (.I0(i2c_addr_s[6]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_12_15_n_3 ),
        .O(\i2c_addr_s[6]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'h00008A00)) 
    \i2c_addr_s[6]_i_3 
       (.I0(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I1(\i2c_addr_s[6]_i_4_n_0 ),
        .I2(\u_i2c_master/st_wr_mode ),
        .I3(\i2c_addr_s[6]_i_5_n_0 ),
        .I4(\u_i2c_master/st_addr_reg_n_0_[2] ),
        .O(\i2c_addr_s[6]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hEFFEFFFFFFFFEFFE)) 
    \i2c_addr_s[6]_i_4 
       (.I0(\i2c_addr_s[6]_i_6_n_0 ),
        .I1(\i2c_addr_s[6]_i_7_n_0 ),
        .I2(\u_i2c_master/st_is_dirty0 [6]),
        .I3(\u_i2c_master/st_reg_data [6]),
        .I4(\u_i2c_master/st_is_dirty0 [7]),
        .I5(\u_i2c_master/st_reg_data [7]),
        .O(\i2c_addr_s[6]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    \i2c_addr_s[6]_i_5 
       (.I0(\u_i2c_master/prog_cache_entries [3]),
        .I1(\u_i2c_master/prog_cache_entries [2]),
        .I2(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .I3(\u_i2c_master/prog_cache_entries [1]),
        .I4(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .I5(\u_i2c_master/prog_cache_entries [0]),
        .O(\i2c_addr_s[6]_i_5_n_0 ));
  LUT6 #(
    .INIT(64'h6FF6FFFFFFFF6FF6)) 
    \i2c_addr_s[6]_i_6 
       (.I0(\u_i2c_master/st_reg_data [0]),
        .I1(\u_i2c_master/st_is_dirty0 [0]),
        .I2(\u_i2c_master/st_is_dirty0 [1]),
        .I3(\u_i2c_master/st_reg_data [1]),
        .I4(\u_i2c_master/st_is_dirty0 [2]),
        .I5(\u_i2c_master/st_reg_data [2]),
        .O(\i2c_addr_s[6]_i_6_n_0 ));
  LUT6 #(
    .INIT(64'h6FF6FFFFFFFF6FF6)) 
    \i2c_addr_s[6]_i_7 
       (.I0(\u_i2c_master/st_reg_data [3]),
        .I1(\u_i2c_master/st_is_dirty0 [3]),
        .I2(\u_i2c_master/st_is_dirty0 [5]),
        .I3(\u_i2c_master/st_reg_data [5]),
        .I4(\u_i2c_master/st_is_dirty0 [4]),
        .I5(\u_i2c_master/st_reg_data [4]),
        .O(\i2c_addr_s[6]_i_7_n_0 ));
  FDRE \i2c_addr_s_reg[0] 
       (.C(clk),
        .CE(i2c_addr_s0),
        .D(dati[0]),
        .Q(i2c_addr_s[0]),
        .R(rst));
  FDRE \i2c_addr_s_reg[1] 
       (.C(clk),
        .CE(i2c_addr_s0),
        .D(dati[1]),
        .Q(i2c_addr_s[1]),
        .R(rst));
  FDRE \i2c_addr_s_reg[2] 
       (.C(clk),
        .CE(i2c_addr_s0),
        .D(dati[2]),
        .Q(i2c_addr_s[2]),
        .R(rst));
  FDRE \i2c_addr_s_reg[3] 
       (.C(clk),
        .CE(i2c_addr_s0),
        .D(dati[3]),
        .Q(i2c_addr_s[3]),
        .R(rst));
  FDRE \i2c_addr_s_reg[4] 
       (.C(clk),
        .CE(i2c_addr_s0),
        .D(dati[4]),
        .Q(i2c_addr_s[4]),
        .R(rst));
  FDRE \i2c_addr_s_reg[5] 
       (.C(clk),
        .CE(i2c_addr_s0),
        .D(dati[5]),
        .Q(i2c_addr_s[5]),
        .R(rst));
  FDRE \i2c_addr_s_reg[6] 
       (.C(clk),
        .CE(i2c_addr_s0),
        .D(dati[6]),
        .Q(i2c_addr_s[6]),
        .R(rst));
  LUT6 #(
    .INIT(64'h0000000040000000)) 
    \i2c_cache_addr[0]_i_1 
       (.I0(i2c_cache_addr[0]),
        .I1(rfmod_id[1]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[0]),
        .I4(rfmod_id[3]),
        .I5(rst),
        .O(\i2c_cache_addr[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT4 #(
    .INIT(16'h0060)) 
    \i2c_cache_addr[1]_i_1 
       (.I0(i2c_cache_addr[0]),
        .I1(i2c_cache_addr[1]),
        .I2(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I3(rst),
        .O(\i2c_cache_addr[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair30" *) 
  LUT5 #(
    .INIT(32'h00007800)) 
    \i2c_cache_addr[2]_i_1 
       (.I0(i2c_cache_addr[1]),
        .I1(i2c_cache_addr[0]),
        .I2(i2c_cache_addr[2]),
        .I3(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I4(rst),
        .O(\i2c_cache_addr[2]_i_1_n_0 ));
  FDRE \i2c_cache_addr_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\i2c_cache_addr[0]_i_1_n_0 ),
        .Q(i2c_cache_addr[0]),
        .R(1'b0));
  FDRE \i2c_cache_addr_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\i2c_cache_addr[1]_i_1_n_0 ),
        .Q(i2c_cache_addr[1]),
        .R(1'b0));
  FDRE \i2c_cache_addr_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\i2c_cache_addr[2]_i_1_n_0 ),
        .Q(i2c_cache_addr[2]),
        .R(1'b0));
  LUT2 #(
    .INIT(4'h8)) 
    i2c_cmd_en_i_1
       (.I0(rdy),
        .I1(i2c_addr_s0),
        .O(i2c_cmd_en0));
  FDRE i2c_cmd_en_reg
       (.C(clk),
        .CE(1'b1),
        .D(i2c_cmd_en0),
        .Q(i2c_cmd_en),
        .R(1'b0));
  (* SOFT_HLUTNM = "soft_lutpair121" *) 
  LUT3 #(
    .INIT(8'h8B)) 
    \i2c_cmd_mode[0]_i_1 
       (.I0(i2c_cmd_mode[0]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_wr_mode ),
        .O(\u_i2c_master/i2c_cmd_mode [0]));
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_cmd_mode[1]_i_1 
       (.I0(i2c_cmd_mode[1]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_wr_mode ),
        .O(\u_i2c_master/i2c_cmd_mode [1]));
  (* SOFT_HLUTNM = "soft_lutpair136" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_cmd_mode[2]_i_1 
       (.I0(i2c_cmd_mode[2]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_wr_mode ),
        .O(\u_i2c_master/i2c_cmd_mode [2]));
  (* SOFT_HLUTNM = "soft_lutpair121" *) 
  LUT3 #(
    .INIT(8'h8B)) 
    \i2c_cmd_mode[3]_i_1 
       (.I0(i2c_cmd_mode[3]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_wr_mode ),
        .O(\u_i2c_master/i2c_cmd_mode [3]));
  FDRE \i2c_cmd_mode_reg[0] 
       (.C(clk),
        .CE(i2c_addr_s0),
        .D(dati[7]),
        .Q(i2c_cmd_mode[0]),
        .R(rst));
  FDRE \i2c_cmd_mode_reg[1] 
       (.C(clk),
        .CE(i2c_addr_s0),
        .D(dati[8]),
        .Q(i2c_cmd_mode[1]),
        .R(rst));
  FDRE \i2c_cmd_mode_reg[2] 
       (.C(clk),
        .CE(i2c_addr_s0),
        .D(dati[9]),
        .Q(i2c_cmd_mode[2]),
        .R(rst));
  FDRE \i2c_cmd_mode_reg[3] 
       (.C(clk),
        .CE(i2c_addr_s0),
        .D(dati[10]),
        .Q(i2c_cmd_mode[3]),
        .R(rst));
  (* SOFT_HLUTNM = "soft_lutpair104" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i0[0]_i_1 
       (.I0(i2c_data_i0[0]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_0_5_n_1 ),
        .O(\i2c_data_i0[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair109" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i0[1]_i_1 
       (.I0(i2c_data_i0[1]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_0_5_n_0 ),
        .O(\i2c_data_i0[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair109" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i0[2]_i_1 
       (.I0(i2c_data_i0[2]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_0_5_n_3 ),
        .O(\i2c_data_i0[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair110" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i0[3]_i_1 
       (.I0(i2c_data_i0[3]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_0_5_n_2 ),
        .O(\i2c_data_i0[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair110" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i0[4]_i_1 
       (.I0(i2c_data_i0[4]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_0_5_n_5 ),
        .O(\i2c_data_i0[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair111" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i0[5]_i_1 
       (.I0(i2c_data_i0[5]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_0_5_n_4 ),
        .O(\i2c_data_i0[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair111" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i0[6]_i_1 
       (.I0(i2c_data_i0[6]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_6_11_n_1 ),
        .O(\i2c_data_i0[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair112" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i0[7]_i_1 
       (.I0(i2c_data_i0[7]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/prog_cache_reg_0_7_6_11_n_0 ),
        .O(\i2c_data_i0[7]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000000100000)) 
    \i2c_data_i0[7]_i_1__0 
       (.I0(addr[2]),
        .I1(addr[5]),
        .I2(addr[3]),
        .I3(addr[4]),
        .I4(\idovr[4]_i_2_n_0 ),
        .I5(addr[1]),
        .O(__do_out1103_out));
  FDRE \i2c_data_i0_reg[0] 
       (.C(clk),
        .CE(__do_out1103_out),
        .D(dati[0]),
        .Q(i2c_data_i0[0]),
        .R(rst));
  FDRE \i2c_data_i0_reg[1] 
       (.C(clk),
        .CE(__do_out1103_out),
        .D(dati[1]),
        .Q(i2c_data_i0[1]),
        .R(rst));
  FDRE \i2c_data_i0_reg[2] 
       (.C(clk),
        .CE(__do_out1103_out),
        .D(dati[2]),
        .Q(i2c_data_i0[2]),
        .R(rst));
  FDRE \i2c_data_i0_reg[3] 
       (.C(clk),
        .CE(__do_out1103_out),
        .D(dati[3]),
        .Q(i2c_data_i0[3]),
        .R(rst));
  FDRE \i2c_data_i0_reg[4] 
       (.C(clk),
        .CE(__do_out1103_out),
        .D(dati[4]),
        .Q(i2c_data_i0[4]),
        .R(rst));
  FDRE \i2c_data_i0_reg[5] 
       (.C(clk),
        .CE(__do_out1103_out),
        .D(dati[5]),
        .Q(i2c_data_i0[5]),
        .R(rst));
  FDRE \i2c_data_i0_reg[6] 
       (.C(clk),
        .CE(__do_out1103_out),
        .D(dati[6]),
        .Q(i2c_data_i0[6]),
        .R(rst));
  FDRE \i2c_data_i0_reg[7] 
       (.C(clk),
        .CE(__do_out1103_out),
        .D(dati[7]),
        .Q(i2c_data_i0[7]),
        .R(rst));
  (* SOFT_HLUTNM = "soft_lutpair112" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i1[0]_i_1 
       (.I0(i2c_data_i1[0]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_reg_data [0]),
        .O(\i2c_data_i1[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair113" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i1[1]_i_1 
       (.I0(i2c_data_i1[1]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_reg_data [1]),
        .O(\i2c_data_i1[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair113" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i1[2]_i_1 
       (.I0(i2c_data_i1[2]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_reg_data [2]),
        .O(\i2c_data_i1[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair131" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i1[3]_i_1 
       (.I0(i2c_data_i1[3]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_reg_data [3]),
        .O(\i2c_data_i1[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair132" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i1[4]_i_1 
       (.I0(i2c_data_i1[4]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_reg_data [4]),
        .O(\i2c_data_i1[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair133" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i1[5]_i_1 
       (.I0(i2c_data_i1[5]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_reg_data [5]),
        .O(\i2c_data_i1[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair134" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i1[6]_i_1 
       (.I0(i2c_data_i1[6]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_reg_data [6]),
        .O(\i2c_data_i1[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair104" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \i2c_data_i1[7]_i_1 
       (.I0(i2c_data_i1[7]),
        .I1(i2c_busy),
        .I2(\u_i2c_master/st_reg_data [7]),
        .O(\i2c_data_i1[7]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h00100000)) 
    \i2c_data_i1[7]_i_1__0 
       (.I0(addr[2]),
        .I1(addr[5]),
        .I2(addr[3]),
        .I3(addr[4]),
        .I4(\idovr[4]_i_2_n_0 ),
        .O(\i2c_data_i1[7]_i_1__0_n_0 ));
  FDRE \i2c_data_i1_reg[0] 
       (.C(clk),
        .CE(\i2c_data_i1[7]_i_1__0_n_0 ),
        .D(dati[8]),
        .Q(i2c_data_i1[0]),
        .R(rst));
  FDRE \i2c_data_i1_reg[1] 
       (.C(clk),
        .CE(\i2c_data_i1[7]_i_1__0_n_0 ),
        .D(dati[9]),
        .Q(i2c_data_i1[1]),
        .R(rst));
  FDRE \i2c_data_i1_reg[2] 
       (.C(clk),
        .CE(\i2c_data_i1[7]_i_1__0_n_0 ),
        .D(dati[10]),
        .Q(i2c_data_i1[2]),
        .R(rst));
  FDRE \i2c_data_i1_reg[3] 
       (.C(clk),
        .CE(\i2c_data_i1[7]_i_1__0_n_0 ),
        .D(dati[11]),
        .Q(i2c_data_i1[3]),
        .R(rst));
  FDRE \i2c_data_i1_reg[4] 
       (.C(clk),
        .CE(\i2c_data_i1[7]_i_1__0_n_0 ),
        .D(dati[12]),
        .Q(i2c_data_i1[4]),
        .R(rst));
  FDRE \i2c_data_i1_reg[5] 
       (.C(clk),
        .CE(\i2c_data_i1[7]_i_1__0_n_0 ),
        .D(dati[13]),
        .Q(i2c_data_i1[5]),
        .R(rst));
  FDRE \i2c_data_i1_reg[6] 
       (.C(clk),
        .CE(\i2c_data_i1[7]_i_1__0_n_0 ),
        .D(dati[14]),
        .Q(i2c_data_i1[6]),
        .R(rst));
  FDRE \i2c_data_i1_reg[7] 
       (.C(clk),
        .CE(\i2c_data_i1[7]_i_1__0_n_0 ),
        .D(dati[15]),
        .Q(i2c_data_i1[7]),
        .R(rst));
  LUT4 #(
    .INIT(16'h0002)) 
    \i2c_data_i2[7]_i_1 
       (.I0(i2c_busy),
        .I1(\u_i2c_master/state_reg_n_0_[0] ),
        .I2(\u_i2c_master/state_reg_n_0_[2] ),
        .I3(\u_i2c_master/state_reg_n_0_[1] ),
        .O(\i2c_data_i2[7]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0010000000000000)) 
    \i2c_data_i2[7]_i_1__0 
       (.I0(addr[2]),
        .I1(addr[5]),
        .I2(addr[3]),
        .I3(addr[4]),
        .I4(\idovr[4]_i_2_n_0 ),
        .I5(addr[1]),
        .O(\i2c_data_i2[7]_i_1__0_n_0 ));
  FDRE \i2c_data_i2_reg[0] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1__0_n_0 ),
        .D(dati[0]),
        .Q(i2c_data_i2[0]),
        .R(rst));
  FDRE \i2c_data_i2_reg[1] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1__0_n_0 ),
        .D(dati[1]),
        .Q(i2c_data_i2[1]),
        .R(rst));
  FDRE \i2c_data_i2_reg[2] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1__0_n_0 ),
        .D(dati[2]),
        .Q(i2c_data_i2[2]),
        .R(rst));
  FDRE \i2c_data_i2_reg[3] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1__0_n_0 ),
        .D(dati[3]),
        .Q(i2c_data_i2[3]),
        .R(rst));
  FDRE \i2c_data_i2_reg[4] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1__0_n_0 ),
        .D(dati[4]),
        .Q(i2c_data_i2[4]),
        .R(rst));
  FDRE \i2c_data_i2_reg[5] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1__0_n_0 ),
        .D(dati[5]),
        .Q(i2c_data_i2[5]),
        .R(rst));
  FDRE \i2c_data_i2_reg[6] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1__0_n_0 ),
        .D(dati[6]),
        .Q(i2c_data_i2[6]),
        .R(rst));
  FDRE \i2c_data_i2_reg[7] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1__0_n_0 ),
        .D(dati[7]),
        .Q(i2c_data_i2[7]),
        .R(rst));
  LUT5 #(
    .INIT(32'h00900000)) 
    \i2c_prog_cache_addr[0]_i_1 
       (.I0(i2c_prog_cache_addr[0]),
        .I1(i2c_prog_cache_addr[2]),
        .I2(dut_pgood),
        .I3(rst),
        .I4(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .O(\i2c_prog_cache_addr[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h00009A0000000000)) 
    \i2c_prog_cache_addr[1]_i_1 
       (.I0(i2c_prog_cache_addr[1]),
        .I1(i2c_prog_cache_addr[2]),
        .I2(i2c_prog_cache_addr[0]),
        .I3(dut_pgood),
        .I4(rst),
        .I5(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .O(\i2c_prog_cache_addr[1]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000EA0000000000)) 
    \i2c_prog_cache_addr[2]_i_1 
       (.I0(i2c_prog_cache_addr[2]),
        .I1(i2c_prog_cache_addr[1]),
        .I2(i2c_prog_cache_addr[0]),
        .I3(dut_pgood),
        .I4(rst),
        .I5(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .O(\i2c_prog_cache_addr[2]_i_1_n_0 ));
  FDRE \i2c_prog_cache_addr_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\i2c_prog_cache_addr[0]_i_1_n_0 ),
        .Q(i2c_prog_cache_addr[0]),
        .R(1'b0));
  FDRE \i2c_prog_cache_addr_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\i2c_prog_cache_addr[1]_i_1_n_0 ),
        .Q(i2c_prog_cache_addr[1]),
        .R(1'b0));
  FDRE \i2c_prog_cache_addr_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\i2c_prog_cache_addr[2]_i_1_n_0 ),
        .Q(i2c_prog_cache_addr[2]),
        .R(1'b0));
  LUT6 #(
    .INIT(64'hEFFEFFFFFFFFEFFE)) 
    id_changed_i_1
       (.I0(id_changed),
        .I1(id_changed_i_2_n_0),
        .I2(rfmod_in[4]),
        .I3(rfmod_id[3]),
        .I4(rfmod_in[3]),
        .I5(rfmod_id[2]),
        .O(id_changed0));
  (* SOFT_HLUTNM = "soft_lutpair99" *) 
  LUT4 #(
    .INIT(16'h6FF6)) 
    id_changed_i_2
       (.I0(rfmod_id[1]),
        .I1(rfmod_in[2]),
        .I2(rfmod_id[0]),
        .I3(rfmod_in[1]),
        .O(id_changed_i_2_n_0));
  FDRE id_changed_reg
       (.C(clk),
        .CE(1'b1),
        .D(id_changed0),
        .Q(id_changed),
        .R(rst));
  LUT4 #(
    .INIT(16'h8110)) 
    id_valid_INST_0
       (.I0(rfmod_id[3]),
        .I1(rfmod_id[2]),
        .I2(rfmod_id[0]),
        .I3(rfmod_id[1]),
        .O(id_valid));
  LUT6 #(
    .INIT(64'h0010000000000000)) 
    \idovr[4]_i_1 
       (.I0(addr[5]),
        .I1(addr[3]),
        .I2(addr[2]),
        .I3(addr[4]),
        .I4(addr[1]),
        .I5(\idovr[4]_i_2_n_0 ),
        .O(idovr0));
  (* SOFT_HLUTNM = "soft_lutpair32" *) 
  LUT3 #(
    .INIT(8'h08)) 
    \idovr[4]_i_2 
       (.I0(en),
        .I1(wr),
        .I2(addr[0]),
        .O(\idovr[4]_i_2_n_0 ));
  FDRE \idovr_reg[0] 
       (.C(clk),
        .CE(idovr0),
        .D(dati[0]),
        .Q(idovr[0]),
        .R(rst));
  FDRE \idovr_reg[1] 
       (.C(clk),
        .CE(idovr0),
        .D(dati[1]),
        .Q(idovr[1]),
        .R(rst));
  FDRE \idovr_reg[2] 
       (.C(clk),
        .CE(idovr0),
        .D(dati[2]),
        .Q(idovr[2]),
        .R(rst));
  FDRE \idovr_reg[3] 
       (.C(clk),
        .CE(idovr0),
        .D(dati[3]),
        .Q(idovr[3]),
        .R(rst));
  FDRE \idovr_reg[4] 
       (.C(clk),
        .CE(idovr0),
        .D(dati[4]),
        .Q(idovr[4]),
        .R(rst));
  LUT6 #(
    .INIT(64'h0000020000000000)) 
    \jtag_ctrl[4]_i_1 
       (.I0(\test_ctrl[15]_i_2_n_0 ),
        .I1(addr[5]),
        .I2(addr[3]),
        .I3(addr[2]),
        .I4(addr[4]),
        .I5(addr[1]),
        .O(__do_out1108_out));
  FDRE \jtag_ctrl_reg[0] 
       (.C(clk),
        .CE(__do_out1108_out),
        .D(dati[0]),
        .Q(\jtag_ctrl_reg_n_0_[0] ),
        .R(rst));
  FDRE \jtag_ctrl_reg[1] 
       (.C(clk),
        .CE(__do_out1108_out),
        .D(dati[1]),
        .Q(\jtag_ctrl_reg_n_0_[1] ),
        .R(rst));
  FDRE \jtag_ctrl_reg[2] 
       (.C(clk),
        .CE(__do_out1108_out),
        .D(dati[2]),
        .Q(prog_jdi),
        .R(rst));
  FDRE \jtag_ctrl_reg[4] 
       (.C(clk),
        .CE(__do_out1108_out),
        .D(dati[4]),
        .Q(prog_jen),
        .R(rst));
  LUT6 #(
    .INIT(64'hFFFFBFFF00008000)) 
    last_reg_i_1
       (.I0(\u_i2c_master/data_in_last ),
        .I1(\u_i2c_master/u_i2c_master/data_in_ready0 ),
        .I2(\u_i2c_master/u_i2c_master/data_in_valid0 ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .I4(\addr_reg[6]_i_2_n_0 ),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/last_reg ),
        .O(last_reg_i_1_n_0));
  LUT6 #(
    .INIT(64'h0000004000000000)) 
    last_wr_cache_reg_0_7_0_0_i_1
       (.I0(\u_i2c_master/u_i2c_master/error_reg_n_0 ),
        .I1(\u_i2c_master/st_wr_mode ),
        .I2(\state[1]_i_2_n_0 ),
        .I3(rst),
        .I4(\u_i2c_master/state_reg_n_0_[0] ),
        .I5(last_wr_cache_reg_0_7_0_0_i_2_n_0),
        .O(p_0_in__0));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT2 #(
    .INIT(4'h2)) 
    last_wr_cache_reg_0_7_0_0_i_2
       (.I0(\u_i2c_master/state_reg_n_0_[1] ),
        .I1(\u_i2c_master/state_reg_n_0_[2] ),
        .O(last_wr_cache_reg_0_7_0_0_i_2_n_0));
  LUT2 #(
    .INIT(4'hE)) 
    \mask_count[7]_i_1 
       (.I0(rst),
        .I1(test_stat[29]),
        .O(\u_tester/mask_count ));
  LUT6 #(
    .INIT(64'h0040555555555555)) 
    \mem_read_data_reg[8]_i_1 
       (.I0(\mem_read_data_reg[8]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/state [1]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/u_i2c_master/data_out_valid ),
        .I5(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_valid_reg ),
        .O(\mem_read_data_reg[8]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'h00BF)) 
    \mem_read_data_reg[8]_i_1__0 
       (.I0(\u_i2c_master/u_i2c_master/data_in_ready0 ),
        .I1(\u_i2c_master/u_i2c_master/data_in_valid0 ),
        .I2(\u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_valid_reg ),
        .I3(\mem_read_data_reg[8]_i_2__0_n_0 ),
        .O(\mem_read_data_reg[8]_i_1__0_n_0 ));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    \mem_read_data_reg[8]_i_2 
       (.I0(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [1]),
        .I1(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[1] ),
        .I2(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [0]),
        .I3(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[0] ),
        .I4(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[2] ),
        .I5(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [2]),
        .O(\mem_read_data_reg[8]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h9009000000009009)) 
    \mem_read_data_reg[8]_i_2__0 
       (.I0(\u_i2c_master/rd_ptr_reg [2]),
        .I1(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [2]),
        .I2(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [1]),
        .I3(\u_i2c_master/rd_ptr_reg [1]),
        .I4(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [0]),
        .I5(\u_i2c_master/rd_ptr_reg [0]),
        .O(\mem_read_data_reg[8]_i_2__0_n_0 ));
  LUT6 #(
    .INIT(64'hFF7F555555555555)) 
    mem_read_data_valid_reg_i_1
       (.I0(\mem_read_data_reg[8]_i_2_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/state [1]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/u_i2c_master/data_out_valid ),
        .I5(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_valid_reg ),
        .O(mem_read_data_valid_reg_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair94" *) 
  LUT4 #(
    .INIT(16'h7555)) 
    mem_read_data_valid_reg_i_1__0
       (.I0(\mem_read_data_reg[8]_i_2__0_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/data_in_ready0 ),
        .I2(\u_i2c_master/u_i2c_master/data_in_valid0 ),
        .I3(\u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_valid_reg ),
        .O(mem_read_data_valid_reg_i_1__0_n_0));
  LUT2 #(
    .INIT(4'h2)) 
    mem_reg_0_3_0_5_i_1
       (.I0(\u_i2c_master/u_i2c_master/data_out_valid0 ),
        .I1(\wr_addr_reg[1]_i_2_n_0 ),
        .O(write1_out__0));
  LUT6 #(
    .INIT(64'h0011001000000010)) 
    mem_reg_0_3_0_5_i_10
       (.I0(\u_i2c_master/u_i2c_master/state [3]),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/i2c_data_i1 [1]),
        .I3(\u_i2c_master/u_i2c_master/state [0]),
        .I4(\u_i2c_master/u_i2c_master/state [1]),
        .I5(\u_i2c_master/i2c_data_i2 [1]),
        .O(mem_reg_0_3_0_5_i_10_n_0));
  LUT6 #(
    .INIT(64'h0011001000000010)) 
    mem_reg_0_3_0_5_i_11
       (.I0(\u_i2c_master/u_i2c_master/state [3]),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/i2c_data_i1 [0]),
        .I3(\u_i2c_master/u_i2c_master/state [0]),
        .I4(\u_i2c_master/u_i2c_master/state [1]),
        .I5(\u_i2c_master/i2c_data_i2 [0]),
        .O(mem_reg_0_3_0_5_i_11_n_0));
  LUT6 #(
    .INIT(64'h0011001000000010)) 
    mem_reg_0_3_0_5_i_12
       (.I0(\u_i2c_master/u_i2c_master/state [3]),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/i2c_data_i1 [3]),
        .I3(\u_i2c_master/u_i2c_master/state [0]),
        .I4(\u_i2c_master/u_i2c_master/state [1]),
        .I5(\u_i2c_master/i2c_data_i2 [3]),
        .O(mem_reg_0_3_0_5_i_12_n_0));
  LUT6 #(
    .INIT(64'h0011001000000010)) 
    mem_reg_0_3_0_5_i_13
       (.I0(\u_i2c_master/u_i2c_master/state [3]),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/i2c_data_i1 [2]),
        .I3(\u_i2c_master/u_i2c_master/state [0]),
        .I4(\u_i2c_master/u_i2c_master/state [1]),
        .I5(\u_i2c_master/i2c_data_i2 [2]),
        .O(mem_reg_0_3_0_5_i_13_n_0));
  LUT6 #(
    .INIT(64'h0011001000000010)) 
    mem_reg_0_3_0_5_i_14
       (.I0(\u_i2c_master/u_i2c_master/state [3]),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/i2c_data_i1 [5]),
        .I3(\u_i2c_master/u_i2c_master/state [0]),
        .I4(\u_i2c_master/u_i2c_master/state [1]),
        .I5(\u_i2c_master/i2c_data_i2 [5]),
        .O(mem_reg_0_3_0_5_i_14_n_0));
  LUT6 #(
    .INIT(64'h0011001000000010)) 
    mem_reg_0_3_0_5_i_15
       (.I0(\u_i2c_master/u_i2c_master/state [3]),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/i2c_data_i1 [4]),
        .I3(\u_i2c_master/u_i2c_master/state [0]),
        .I4(\u_i2c_master/u_i2c_master/state [1]),
        .I5(\u_i2c_master/i2c_data_i2 [4]),
        .O(mem_reg_0_3_0_5_i_15_n_0));
  LUT6 #(
    .INIT(64'hBBB0B0BB00000000)) 
    mem_reg_0_3_0_5_i_1__0
       (.I0(\u_i2c_master/u_i2c_master/state [1]),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(mem_reg_0_3_0_5_i_8_n_0),
        .I3(\u_i2c_master/rd_ptr_reg [2]),
        .I4(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [2]),
        .I5(mem_reg_0_3_0_5_i_9_n_0),
        .O(mem_reg_0_3_0_5_i_1__0_n_0));
  LUT5 #(
    .INIT(32'hFFFEAAAA)) 
    mem_reg_0_3_0_5_i_2
       (.I0(mem_reg_0_3_0_5_i_10_n_0),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/i2c_data_i0 [1]),
        .O(mem_reg_0_3_0_5_i_2_n_0));
  LUT5 #(
    .INIT(32'hFFFEAAAA)) 
    mem_reg_0_3_0_5_i_3
       (.I0(mem_reg_0_3_0_5_i_11_n_0),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/i2c_data_i0 [0]),
        .O(mem_reg_0_3_0_5_i_3_n_0));
  LUT5 #(
    .INIT(32'hFFFEAAAA)) 
    mem_reg_0_3_0_5_i_4
       (.I0(mem_reg_0_3_0_5_i_12_n_0),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/i2c_data_i0 [3]),
        .O(mem_reg_0_3_0_5_i_4_n_0));
  LUT5 #(
    .INIT(32'hFFFEAAAA)) 
    mem_reg_0_3_0_5_i_5
       (.I0(mem_reg_0_3_0_5_i_13_n_0),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/i2c_data_i0 [2]),
        .O(mem_reg_0_3_0_5_i_5_n_0));
  LUT5 #(
    .INIT(32'hFFFEAAAA)) 
    mem_reg_0_3_0_5_i_6
       (.I0(mem_reg_0_3_0_5_i_14_n_0),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/i2c_data_i0 [5]),
        .O(mem_reg_0_3_0_5_i_6_n_0));
  LUT5 #(
    .INIT(32'hFFFEAAAA)) 
    mem_reg_0_3_0_5_i_7
       (.I0(mem_reg_0_3_0_5_i_15_n_0),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/i2c_data_i0 [4]),
        .O(mem_reg_0_3_0_5_i_7_n_0));
  (* SOFT_HLUTNM = "soft_lutpair67" *) 
  LUT4 #(
    .INIT(16'h6FF6)) 
    mem_reg_0_3_0_5_i_8
       (.I0(\u_i2c_master/rd_ptr_reg [0]),
        .I1(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [0]),
        .I2(\u_i2c_master/rd_ptr_reg [1]),
        .I3(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [1]),
        .O(mem_reg_0_3_0_5_i_8_n_0));
  (* SOFT_HLUTNM = "soft_lutpair1" *) 
  LUT2 #(
    .INIT(4'h1)) 
    mem_reg_0_3_0_5_i_9
       (.I0(\u_i2c_master/u_i2c_master/state [2]),
        .I1(\u_i2c_master/u_i2c_master/state [3]),
        .O(mem_reg_0_3_0_5_i_9_n_0));
  LUT5 #(
    .INIT(32'hFFFEAAAA)) 
    mem_reg_0_3_6_8_i_1
       (.I0(mem_reg_0_3_6_8_i_4_n_0),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/i2c_data_i0 [7]),
        .O(mem_reg_0_3_6_8_i_1_n_0));
  LUT5 #(
    .INIT(32'hFFFEAAAA)) 
    mem_reg_0_3_6_8_i_2
       (.I0(mem_reg_0_3_6_8_i_5_n_0),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/i2c_data_i0 [6]),
        .O(mem_reg_0_3_6_8_i_2_n_0));
  LUT5 #(
    .INIT(32'hFFFFBA00)) 
    mem_reg_0_3_6_8_i_3
       (.I0(\u_i2c_master/u_i2c_master/state [1]),
        .I1(\u_i2c_master/u_i2c_master/cmd_mode_r_reg_n_0_[0] ),
        .I2(\u_i2c_master/u_i2c_master/p_0_in7_in ),
        .I3(mem_reg_0_3_6_8_i_6_n_0),
        .I4(mem_reg_0_3_6_8_i_7_n_0),
        .O(\u_i2c_master/u_i2c_master/data_in_last ));
  LUT6 #(
    .INIT(64'h0011001000000010)) 
    mem_reg_0_3_6_8_i_4
       (.I0(\u_i2c_master/u_i2c_master/state [3]),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/i2c_data_i1 [7]),
        .I3(\u_i2c_master/u_i2c_master/state [0]),
        .I4(\u_i2c_master/u_i2c_master/state [1]),
        .I5(\u_i2c_master/i2c_data_i2 [7]),
        .O(mem_reg_0_3_6_8_i_4_n_0));
  LUT6 #(
    .INIT(64'h0011001000000010)) 
    mem_reg_0_3_6_8_i_5
       (.I0(\u_i2c_master/u_i2c_master/state [3]),
        .I1(\u_i2c_master/u_i2c_master/state [2]),
        .I2(\u_i2c_master/i2c_data_i1 [6]),
        .I3(\u_i2c_master/u_i2c_master/state [0]),
        .I4(\u_i2c_master/u_i2c_master/state [1]),
        .I5(\u_i2c_master/i2c_data_i2 [6]),
        .O(mem_reg_0_3_6_8_i_5_n_0));
  (* SOFT_HLUTNM = "soft_lutpair103" *) 
  LUT3 #(
    .INIT(8'h01)) 
    mem_reg_0_3_6_8_i_6
       (.I0(\u_i2c_master/u_i2c_master/state [0]),
        .I1(\u_i2c_master/u_i2c_master/state [3]),
        .I2(\u_i2c_master/u_i2c_master/state [2]),
        .O(mem_reg_0_3_6_8_i_6_n_0));
  LUT6 #(
    .INIT(64'h8080808000008000)) 
    mem_reg_0_3_6_8_i_7
       (.I0(\u_i2c_master/u_i2c_master/state [1]),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(mem_reg_0_3_0_5_i_9_n_0),
        .I3(\u_i2c_master/u_i2c_master/p_1_in ),
        .I4(\u_i2c_master/u_i2c_master/p_0_in7_in ),
        .I5(\u_i2c_master/u_i2c_master/p_4_in ),
        .O(mem_reg_0_3_6_8_i_7_n_0));
  LUT5 #(
    .INIT(32'h000000E0)) 
    missed_ack_reg_i_1
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[3] ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[10] ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [0]),
        .I3(\addr_reg[6]_i_2_n_0 ),
        .I4(rst),
        .O(missed_ack_reg_i_1_n_0));
  LUT4 #(
    .INIT(16'hFB08)) 
    mode_ping_reg_i_1
       (.I0(\u_i2c_master/u_i2c_master/cmd_ping09_out ),
        .I1(\FSM_onehot_state_reg[1]_i_2_n_0 ),
        .I2(\addr_reg[6]_i_2_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/mode_ping_reg ),
        .O(mode_ping_reg_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT4 #(
    .INIT(16'h0120)) 
    mode_read_reg_i_1
       (.I0(\u_i2c_master/u_i2c_master/state [0]),
        .I1(\u_i2c_master/u_i2c_master/state [1]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .O(\u_i2c_master/u_i2c_master/cmd_read0 ));
  (* SOFT_HLUTNM = "soft_lutpair102" *) 
  LUT1 #(
    .INIT(2'h1)) 
    mode_stop_reg_i_1
       (.I0(mode_stop_reg_i_2_n_0),
        .O(\u_i2c_master/u_i2c_master/cmd_stop0 ));
  LUT6 #(
    .INIT(64'h2A2A2A2A2A2A2A22)) 
    mode_stop_reg_i_2
       (.I0(mode_stop_reg_i_3_n_0),
        .I1(\FSM_onehot_state_reg[6]_i_5_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/p_4_in ),
        .I3(\u_i2c_master/u_i2c_master/p_0_in7_in ),
        .I4(\u_i2c_master/u_i2c_master/p_1_in ),
        .I5(\u_i2c_master/u_i2c_master/cmd_mode_r_reg_n_0_[0] ),
        .O(mode_stop_reg_i_2_n_0));
  (* SOFT_HLUTNM = "soft_lutpair59" *) 
  LUT5 #(
    .INIT(32'hFFFEFF33)) 
    mode_stop_reg_i_3
       (.I0(\u_i2c_master/u_i2c_master/p_0_in7_in ),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [1]),
        .I4(\u_i2c_master/u_i2c_master/state [2]),
        .O(mode_stop_reg_i_3_n_0));
  LUT6 #(
    .INIT(64'h0010000000000000)) 
    mode_write_multiple_reg_i_1
       (.I0(\u_i2c_master/u_i2c_master/state [1]),
        .I1(\u_i2c_master/u_i2c_master/state [0]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/u_i2c_master/p_1_in ),
        .I5(\u_i2c_master/u_i2c_master/p_0_in7_in ),
        .O(\u_i2c_master/u_i2c_master/cmd_write_multiple0 ));
  LUT4 #(
    .INIT(16'h9669)) 
    \out[0]_i_1 
       (.I0(\u_tester/chk_out [7]),
        .I1(\u_tester/chk_out [3]),
        .I2(\u_tester/chk_out [4]),
        .I3(\u_tester/chk_out [5]),
        .O(\out[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair100" *) 
  LUT4 #(
    .INIT(16'h9669)) 
    \out[0]_i_1__0 
       (.I0(\u_tester/u_gen/out_reg_n_0_[4] ),
        .I1(\u_tester/u_gen/out_reg_n_0_[5] ),
        .I2(\u_tester/u_gen/out_reg_n_0_[7] ),
        .I3(\u_tester/u_gen/out_reg_n_0_[3] ),
        .O(\u_tester/p_0_out ));
  LUT3 #(
    .INIT(8'hFB)) 
    \out[7]_i_1 
       (.I0(test_stat[29]),
        .I1(test_en),
        .I2(rst),
        .O(\out[7]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \out[7]_i_2 
       (.I0(\u_tester/st_cur [0]),
        .I1(\u_tester/st_cur [1]),
        .O(\u_tester/chk_en ));
  LUT4 #(
    .INIT(16'h08FF)) 
    \output_axis_reg[8]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/state [1]),
        .I1(\u_i2c_master/u_i2c_master/state [3]),
        .I2(\u_i2c_master/u_i2c_master/state [2]),
        .I3(\u_i2c_master/u_i2c_master/data_out_valid ),
        .O(\output_axis_reg[8]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'hB)) 
    \output_axis_reg[8]_i_1__0 
       (.I0(\u_i2c_master/u_i2c_master/data_in_ready0 ),
        .I1(\u_i2c_master/u_i2c_master/data_in_valid0 ),
        .O(\output_axis_reg[8]_i_1__0_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair54" *) 
  LUT5 #(
    .INIT(32'hFFBFAAAA)) 
    output_axis_tvalid_reg_i_1
       (.I0(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_valid_reg ),
        .I1(\u_i2c_master/u_i2c_master/state [1]),
        .I2(\u_i2c_master/u_i2c_master/state [3]),
        .I3(\u_i2c_master/u_i2c_master/state [2]),
        .I4(\u_i2c_master/u_i2c_master/data_out_valid ),
        .O(output_axis_tvalid_reg_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair94" *) 
  LUT3 #(
    .INIT(8'hBA)) 
    output_axis_tvalid_reg_i_1__0
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_valid_reg ),
        .I1(\u_i2c_master/u_i2c_master/data_in_ready0 ),
        .I2(\u_i2c_master/u_i2c_master/data_in_valid0 ),
        .O(output_axis_tvalid_reg_i_1__0_n_0));
  LUT5 #(
    .INIT(32'hFFBF0080)) 
    phy_rx_data_reg_i_1
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/sda_i_reg ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_next ),
        .I2(phy_rx_data_reg_i_2_n_0),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [0]),
        .O(phy_rx_data_reg_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair7" *) 
  LUT3 #(
    .INIT(8'h40)) 
    phy_rx_data_reg_i_2
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .O(phy_rx_data_reg_i_2_n_0));
  LUT5 #(
    .INIT(32'hFFFF0100)) 
    \prog_cache_entries[0]_i_1 
       (.I0(i2c_prog_cache_addr[2]),
        .I1(i2c_prog_cache_addr[1]),
        .I2(i2c_prog_cache_addr[0]),
        .I3(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I4(\u_i2c_master/prog_cache_entries [0]),
        .O(\prog_cache_entries[0]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFFF0400)) 
    \prog_cache_entries[1]_i_1 
       (.I0(i2c_prog_cache_addr[1]),
        .I1(i2c_prog_cache_addr[0]),
        .I2(i2c_prog_cache_addr[2]),
        .I3(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I4(\u_i2c_master/prog_cache_entries [1]),
        .O(\prog_cache_entries[1]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFFF0200)) 
    \prog_cache_entries[2]_i_1 
       (.I0(i2c_prog_cache_addr[1]),
        .I1(i2c_prog_cache_addr[2]),
        .I2(i2c_prog_cache_addr[0]),
        .I3(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I4(\u_i2c_master/prog_cache_entries [2]),
        .O(\prog_cache_entries[2]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFFF0800)) 
    \prog_cache_entries[3]_i_1 
       (.I0(i2c_prog_cache_addr[0]),
        .I1(i2c_prog_cache_addr[1]),
        .I2(i2c_prog_cache_addr[2]),
        .I3(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I4(\u_i2c_master/prog_cache_entries [3]),
        .O(\prog_cache_entries[3]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000000040000000)) 
    prog_cache_reg_0_7_0_5_i_1
       (.I0(i2c_prog_cache_addr[2]),
        .I1(rfmod_id[1]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[0]),
        .I4(rfmod_id[3]),
        .I5(rst),
        .O(prog_cache_reg_0_7_0_5_i_1_n_0));
  LUT2 #(
    .INIT(4'h1)) 
    prog_cache_reg_0_7_0_5_i_2
       (.I0(i2c_prog_cache_addr[2]),
        .I1(i2c_prog_cache_addr[1]),
        .O(prog_cache_reg_0_7_0_5_i_2_n_0));
  LUT2 #(
    .INIT(4'h2)) 
    prog_cache_reg_0_7_0_5_i_3
       (.I0(i2c_prog_cache_addr[0]),
        .I1(i2c_prog_cache_addr[2]),
        .O(prog_cache_reg_0_7_0_5_i_3_n_0));
  LUT2 #(
    .INIT(4'h2)) 
    prog_cache_reg_0_7_0_5_i_4
       (.I0(i2c_prog_cache_addr[1]),
        .I1(i2c_prog_cache_addr[2]),
        .O(prog_cache_reg_0_7_0_5_i_4_n_0));
  LUT1 #(
    .INIT(2'h1)) 
    prog_cache_reg_0_7_12_15_i_1
       (.I0(i2c_prog_cache_addr[2]),
        .O(prog_cache_reg_0_7_12_15_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair175" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[0]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(rfmod_in[5]),
        .O(\r_inbus[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair146" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[10]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [10]),
        .O(\r_inbus[10]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair147" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[11]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [11]),
        .O(\r_inbus[11]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair148" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[12]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [12]),
        .O(\r_inbus[12]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair150" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[13]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [13]),
        .O(\r_inbus[13]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair151" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[14]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [14]),
        .O(\r_inbus[14]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair152" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[15]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [15]),
        .O(\r_inbus[15]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair154" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[16]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [16]),
        .O(\r_inbus[16]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair155" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[17]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [17]),
        .O(\r_inbus[17]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair153" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[18]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [18]),
        .O(\r_inbus[18]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair162" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[19]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [19]),
        .O(\r_inbus[19]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair178" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[1]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [1]),
        .O(\r_inbus[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair163" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[20]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [20]),
        .O(\r_inbus[20]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair162" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[21]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [21]),
        .O(\r_inbus[21]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair155" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[22]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [22]),
        .O(\r_inbus[22]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair154" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[23]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [23]),
        .O(\r_inbus[23]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair153" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[24]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [24]),
        .O(\r_inbus[24]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair152" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[25]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [25]),
        .O(\r_inbus[25]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair151" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[26]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [26]),
        .O(\r_inbus[26]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair150" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[27]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [27]),
        .O(\r_inbus[27]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair148" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[28]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [28]),
        .O(\r_inbus[28]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair147" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[29]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [29]),
        .O(\r_inbus[29]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair174" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[2]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [2]),
        .O(\r_inbus[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair146" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[30]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [30]),
        .O(\r_inbus[30]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair145" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[31]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [31]),
        .O(\r_inbus[31]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair178" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[3]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [3]),
        .O(\r_inbus[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair174" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[4]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [4]),
        .O(\r_inbus[4]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair172" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[5]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [5]),
        .O(\r_inbus[5]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair172" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[6]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [6]),
        .O(\r_inbus[6]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair163" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[7]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [7]),
        .O(\r_inbus[7]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair144" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[8]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [8]),
        .O(\r_inbus[8]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair145" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \r_inbus[9]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/in15 [9]),
        .O(\r_inbus[9]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \rd_addr_reg[0]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[0] ),
        .I1(\mem_read_data_reg[8]_i_1_n_0 ),
        .O(\rd_addr_reg[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair68" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rd_addr_reg[0]_i_1__0 
       (.I0(\u_i2c_master/rd_ptr_reg [0]),
        .I1(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .O(\rd_addr_reg[0]_i_1__0_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair92" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \rd_addr_reg[1]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[1] ),
        .I1(\mem_read_data_reg[8]_i_1_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[0] ),
        .O(\rd_addr_reg[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair67" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \rd_addr_reg[1]_i_1__0 
       (.I0(\u_i2c_master/rd_ptr_reg [1]),
        .I1(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .I2(\u_i2c_master/rd_ptr_reg [0]),
        .O(\rd_addr_reg[1]_i_1__0_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair92" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \rd_ptr_reg[2]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[2] ),
        .I1(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[0] ),
        .I2(\mem_read_data_reg[8]_i_1_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[1] ),
        .O(\u_i2c_master/rd_ptr_next__0 ));
  (* SOFT_HLUTNM = "soft_lutpair68" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \rd_ptr_reg[2]_i_1__0 
       (.I0(\u_i2c_master/rd_ptr_reg [2]),
        .I1(\u_i2c_master/rd_ptr_reg [0]),
        .I2(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .I3(\u_i2c_master/rd_ptr_reg [1]),
        .O(\u_i2c_master/rd_ptr_next ));
  (* SOFT_HLUTNM = "soft_lutpair106" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \rfmod_id[0]_i_1 
       (.I0(rfmod_in[1]),
        .I1(rst),
        .I2(idovr[0]),
        .O(\rfmod_id[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair108" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \rfmod_id[1]_i_1 
       (.I0(rfmod_in[2]),
        .I1(rst),
        .I2(idovr[1]),
        .O(\rfmod_id[1]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hB8)) 
    \rfmod_id[2]_i_1 
       (.I0(rfmod_in[3]),
        .I1(rst),
        .I2(idovr[2]),
        .O(\rfmod_id[2]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'hE)) 
    \rfmod_id[3]_i_1 
       (.I0(rst),
        .I1(idovr[4]),
        .O(\rfmod_id[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair108" *) 
  LUT3 #(
    .INIT(8'hB8)) 
    \rfmod_id[3]_i_2 
       (.I0(rfmod_in[4]),
        .I1(rst),
        .I2(idovr[3]),
        .O(\rfmod_id[3]_i_2_n_0 ));
  FDRE \rfmod_id_reg[0] 
       (.C(clk),
        .CE(\rfmod_id[3]_i_1_n_0 ),
        .D(\rfmod_id[0]_i_1_n_0 ),
        .Q(rfmod_id[0]),
        .R(1'b0));
  FDRE \rfmod_id_reg[1] 
       (.C(clk),
        .CE(\rfmod_id[3]_i_1_n_0 ),
        .D(\rfmod_id[1]_i_1_n_0 ),
        .Q(rfmod_id[1]),
        .R(1'b0));
  FDRE \rfmod_id_reg[2] 
       (.C(clk),
        .CE(\rfmod_id[3]_i_1_n_0 ),
        .D(\rfmod_id[2]_i_1_n_0 ),
        .Q(rfmod_id[2]),
        .R(1'b0));
  FDRE \rfmod_id_reg[3] 
       (.C(clk),
        .CE(\rfmod_id[3]_i_1_n_0 ),
        .D(\rfmod_id[3]_i_2_n_0 ),
        .Q(rfmod_id[3]),
        .R(1'b0));
  LUT6 #(
    .INIT(64'hCF47CF47FF47CF47)) 
    \rfmod_oe[10]_INST_0 
       (.I0(test_dir),
        .I1(test_en),
        .I2(sda_t),
        .I3(\gpio_dir_reg_n_0_[10] ),
        .I4(dut_pgood),
        .I5(rst),
        .O(rfmod_oe[10]));
  LUT6 #(
    .INIT(64'hCF47CF47FF47CF47)) 
    \rfmod_oe[11]_INST_0 
       (.I0(test_dir),
        .I1(test_en),
        .I2(scl_t),
        .I3(\gpio_dir_reg_n_0_[11] ),
        .I4(dut_pgood),
        .I5(rst),
        .O(rfmod_oe[11]));
  LUT6 #(
    .INIT(64'h5555555511550145)) 
    \rfmod_oe[12]_INST_0 
       (.I0(\rfmod_oe[13]_INST_0_i_1_n_0 ),
        .I1(test_en),
        .I2(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I3(test_dir),
        .I4(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I5(\gpio_dir_reg_n_0_[12] ),
        .O(rfmod_oe[12]));
  LUT6 #(
    .INIT(64'h5555555544445505)) 
    \rfmod_oe[13]_INST_0 
       (.I0(\rfmod_oe[13]_INST_0_i_1_n_0 ),
        .I1(test_dir),
        .I2(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I3(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I4(test_en),
        .I5(\gpio_dir_reg_n_0_[13] ),
        .O(rfmod_oe[13]));
  (* SOFT_HLUTNM = "soft_lutpair97" *) 
  LUT3 #(
    .INIT(8'h45)) 
    \rfmod_oe[13]_INST_0_i_1 
       (.I0(test_en),
        .I1(rst),
        .I2(dut_pgood),
        .O(\rfmod_oe[13]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT4 #(
    .INIT(16'h8000)) 
    \rfmod_oe[13]_INST_0_i_2 
       (.I0(rfmod_id[1]),
        .I1(rfmod_id[2]),
        .I2(rfmod_id[0]),
        .I3(rfmod_id[3]),
        .O(\rfmod_oe[13]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFF2222FF000202)) 
    \rfmod_oe[14]_INST_0 
       (.I0(dut_pgood),
        .I1(rst),
        .I2(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I3(test_dir),
        .I4(test_en),
        .I5(\gpio_dir_reg_n_0_[14] ),
        .O(rfmod_oe[14]));
  LUT6 #(
    .INIT(64'hFFFF2222FF000202)) 
    \rfmod_oe[15]_INST_0 
       (.I0(dut_pgood),
        .I1(rst),
        .I2(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I3(test_dir),
        .I4(test_en),
        .I5(\gpio_dir_reg_n_0_[15] ),
        .O(rfmod_oe[15]));
  LUT6 #(
    .INIT(64'hFFFF2222FF000202)) 
    \rfmod_oe[16]_INST_0 
       (.I0(dut_pgood),
        .I1(rst),
        .I2(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I3(test_dir),
        .I4(test_en),
        .I5(\gpio_dir_reg_n_0_[16] ),
        .O(rfmod_oe[16]));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT4 #(
    .INIT(16'hFFF9)) 
    \rfmod_oe[16]_INST_0_i_1 
       (.I0(rfmod_id[1]),
        .I1(rfmod_id[0]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[3]),
        .O(\rfmod_oe[16]_INST_0_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hDDDD00C0)) 
    \rfmod_oe[1]_INST_0 
       (.I0(test_dir),
        .I1(\gpio_dir_reg_n_0_[1] ),
        .I2(dut_pgood),
        .I3(rst),
        .I4(test_en),
        .O(rfmod_oe[1]));
  LUT5 #(
    .INIT(32'hDDDD00C0)) 
    \rfmod_oe[2]_INST_0 
       (.I0(test_dir),
        .I1(p_1_in),
        .I2(dut_pgood),
        .I3(rst),
        .I4(test_en),
        .O(rfmod_oe[2]));
  LUT5 #(
    .INIT(32'hDDDD00C0)) 
    \rfmod_oe[3]_INST_0 
       (.I0(test_dir),
        .I1(\gpio_dir_reg_n_0_[3] ),
        .I2(dut_pgood),
        .I3(rst),
        .I4(test_en),
        .O(rfmod_oe[3]));
  LUT5 #(
    .INIT(32'hDDDD00C0)) 
    \rfmod_oe[4]_INST_0 
       (.I0(test_dir),
        .I1(\gpio_dir_reg_n_0_[4] ),
        .I2(dut_pgood),
        .I3(rst),
        .I4(test_en),
        .O(rfmod_oe[4]));
  LUT6 #(
    .INIT(64'hFFF02222FFF02020)) 
    \rfmod_oe[5]_INST_0 
       (.I0(dut_pgood),
        .I1(rst),
        .I2(p_2_in),
        .I3(test_dir),
        .I4(test_en),
        .I5(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .O(rfmod_oe[5]));
  LUT6 #(
    .INIT(64'h5555555544445505)) 
    \rfmod_oe[6]_INST_0 
       (.I0(\rfmod_oe[13]_INST_0_i_1_n_0 ),
        .I1(test_dir),
        .I2(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I3(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I4(test_en),
        .I5(\gpio_dir_reg_n_0_[6] ),
        .O(rfmod_oe[6]));
  LUT6 #(
    .INIT(64'h5555555544445505)) 
    \rfmod_oe[7]_INST_0 
       (.I0(\rfmod_oe[13]_INST_0_i_1_n_0 ),
        .I1(test_dir),
        .I2(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I3(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I4(test_en),
        .I5(\gpio_dir_reg_n_0_[7] ),
        .O(rfmod_oe[7]));
  LUT6 #(
    .INIT(64'h5555555455551110)) 
    \rfmod_oe[8]_INST_0 
       (.I0(\rfmod_oe[13]_INST_0_i_1_n_0 ),
        .I1(test_en),
        .I2(\rfmod_oe[8]_INST_0_i_1_n_0 ),
        .I3(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I4(\gpio_dir_reg_n_0_[8] ),
        .I5(test_dir),
        .O(rfmod_oe[8]));
  (* SOFT_HLUTNM = "soft_lutpair58" *) 
  LUT4 #(
    .INIT(16'h0010)) 
    \rfmod_oe[8]_INST_0_i_1 
       (.I0(rfmod_id[3]),
        .I1(rfmod_id[2]),
        .I2(rfmod_id[0]),
        .I3(rfmod_id[1]),
        .O(\rfmod_oe[8]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hF0F0FFFF22202220)) 
    \rfmod_oe[9]_INST_0 
       (.I0(dut_pgood),
        .I1(rst),
        .I2(\gpio_dir_reg_n_0_[9] ),
        .I3(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I4(test_dir),
        .I5(test_en),
        .O(rfmod_oe[9]));
  LUT6 #(
    .INIT(64'hF404F404FFFFF000)) 
    \rfmod_out[10]_INST_0 
       (.I0(test_dir),
        .I1(\rfmod_out[14]_INST_0_i_1_n_0 ),
        .I2(\gpio_dir_reg_n_0_[10] ),
        .I3(\gpio_dato_reg_n_0_[10] ),
        .I4(sda_t),
        .I5(test_en),
        .O(rfmod_out[10]));
  LUT6 #(
    .INIT(64'hF404F404FFFFF000)) 
    \rfmod_out[11]_INST_0 
       (.I0(test_dir),
        .I1(\rfmod_out[11]_INST_0_i_1_n_0 ),
        .I2(\gpio_dir_reg_n_0_[11] ),
        .I3(\gpio_dato_reg_n_0_[11] ),
        .I4(scl_t),
        .I5(test_en),
        .O(rfmod_out[11]));
  (* SOFT_HLUTNM = "soft_lutpair158" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rfmod_out[11]_INST_0_i_1 
       (.I0(test_stat[23]),
        .I1(\u_tester/u_gen/out_reg_n_0_[6] ),
        .O(\rfmod_out[11]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAAAA0FCFAAAA0000)) 
    \rfmod_out[12]_INST_0 
       (.I0(\gpio_dato_reg_n_0_[12] ),
        .I1(\rfmod_out[12]_INST_0_i_1_n_0 ),
        .I2(test_en),
        .I3(test_dir),
        .I4(\gpio_dir_reg_n_0_[12] ),
        .I5(\rfmod_out[12]_INST_0_i_2_n_0 ),
        .O(rfmod_out[12]));
  (* SOFT_HLUTNM = "soft_lutpair156" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rfmod_out[12]_INST_0_i_1 
       (.I0(\u_tester/u_gen/out_reg_n_0_[7] ),
        .I1(test_stat[23]),
        .O(\rfmod_out[12]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFF4F4F444)) 
    \rfmod_out[12]_INST_0_i_2 
       (.I0(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I1(spi_sclk),
        .I2(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I3(\ctrl1_reg_n_0_[12] ),
        .I4(tx_active[1]),
        .I5(test_en),
        .O(\rfmod_out[12]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFF0000FFF8FFF8)) 
    \rfmod_out[13]_INST_0 
       (.I0(\rfmod_out[16]_INST_0_i_1_n_0 ),
        .I1(\rfmod_out[13]_INST_0_i_1_n_0 ),
        .I2(\rfmod_out[13]_INST_0_i_2_n_0 ),
        .I3(\rfmod_out[13]_INST_0_i_3_n_0 ),
        .I4(\gpio_dato_reg_n_0_[13] ),
        .I5(\gpio_dir_reg_n_0_[13] ),
        .O(rfmod_out[13]));
  (* SOFT_HLUTNM = "soft_lutpair100" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rfmod_out[13]_INST_0_i_1 
       (.I0(\u_tester/u_gen/out_reg_n_0_[4] ),
        .I1(test_stat[23]),
        .O(\rfmod_out[13]_INST_0_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h0000E200)) 
    \rfmod_out[13]_INST_0_i_2 
       (.I0(\ctrl1_reg_n_0_[14] ),
        .I1(trx_auto),
        .I2(dut_pgood),
        .I3(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I4(test_en),
        .O(\rfmod_out[13]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h1111100000001000)) 
    \rfmod_out[13]_INST_0_i_3 
       (.I0(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I1(test_en),
        .I2(dut_tms),
        .I3(dut_ten),
        .I4(prog_jen),
        .I5(\jtag_ctrl_reg_n_0_[0] ),
        .O(\rfmod_out[13]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'hB888BBBBB8888888)) 
    \rfmod_out[14]_INST_0 
       (.I0(\gpio_dato_reg_n_0_[14] ),
        .I1(\gpio_dir_reg_n_0_[14] ),
        .I2(\rfmod_out[14]_INST_0_i_1_n_0 ),
        .I3(test_dir),
        .I4(test_en),
        .I5(\rfmod_out[14]_INST_0_i_2_n_0 ),
        .O(rfmod_out[14]));
  (* SOFT_HLUTNM = "soft_lutpair159" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rfmod_out[14]_INST_0_i_1 
       (.I0(\u_tester/u_gen/out_reg_n_0_[5] ),
        .I1(test_stat[23]),
        .O(\rfmod_out[14]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFF0FFF0FFF0E0E0)) 
    \rfmod_out[14]_INST_0_i_2 
       (.I0(p_0_in0_in),
        .I1(tx_active[0]),
        .I2(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .I3(\rfmod_oe[8]_INST_0_i_1_n_0 ),
        .I4(tx_active[1]),
        .I5(\ctrl1_reg_n_0_[12] ),
        .O(\rfmod_out[14]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair60" *) 
  LUT4 #(
    .INIT(16'h0010)) 
    \rfmod_out[14]_INST_0_i_3 
       (.I0(rfmod_id[3]),
        .I1(rfmod_id[2]),
        .I2(rfmod_id[1]),
        .I3(rfmod_id[0]),
        .O(\rfmod_out[14]_INST_0_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8BB88888BBBBBBBB)) 
    \rfmod_out[15]_INST_0 
       (.I0(\gpio_dato_reg_n_0_[15] ),
        .I1(\gpio_dir_reg_n_0_[15] ),
        .I2(test_stat[23]),
        .I3(\u_tester/u_gen/out_reg_n_0_[6] ),
        .I4(\rfmod_out[16]_INST_0_i_1_n_0 ),
        .I5(\rfmod_out[15]_INST_0_i_1_n_0 ),
        .O(rfmod_out[15]));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFF353F)) 
    \rfmod_out[15]_INST_0_i_1 
       (.I0(dut_ten),
        .I1(prog_jdi),
        .I2(prog_jen),
        .I3(dut_tdi),
        .I4(test_en),
        .I5(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .O(\rfmod_out[15]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h88B8B888BBBBBBBB)) 
    \rfmod_out[16]_INST_0 
       (.I0(\gpio_dato_reg_n_0_[16] ),
        .I1(\gpio_dir_reg_n_0_[16] ),
        .I2(\rfmod_out[16]_INST_0_i_1_n_0 ),
        .I3(\u_tester/u_gen/out_reg_n_0_[7] ),
        .I4(test_stat[23]),
        .I5(\rfmod_out[16]_INST_0_i_2_n_0 ),
        .O(rfmod_out[16]));
  (* SOFT_HLUTNM = "soft_lutpair157" *) 
  LUT2 #(
    .INIT(4'h8)) 
    \rfmod_out[16]_INST_0_i_1 
       (.I0(test_en),
        .I1(test_dir),
        .O(\rfmod_out[16]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFFF353F)) 
    \rfmod_out[16]_INST_0_i_2 
       (.I0(dut_ten),
        .I1(\jtag_ctrl_reg_n_0_[1] ),
        .I2(prog_jen),
        .I3(dut_tck),
        .I4(test_en),
        .I5(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .O(\rfmod_out[16]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hF0F6F0F000060000)) 
    \rfmod_out[1]_INST_0 
       (.I0(test_stat[23]),
        .I1(\u_tester/u_gen/out_reg_n_0_[0] ),
        .I2(\gpio_dir_reg_n_0_[1] ),
        .I3(test_dir),
        .I4(test_en),
        .I5(\gpio_dato_reg_n_0_[1] ),
        .O(rfmod_out[1]));
  LUT6 #(
    .INIT(64'hF0F6F0F000060000)) 
    \rfmod_out[2]_INST_0 
       (.I0(test_stat[23]),
        .I1(\u_tester/u_gen/out_reg_n_0_[1] ),
        .I2(p_1_in),
        .I3(test_dir),
        .I4(test_en),
        .I5(\gpio_dato_reg_n_0_[2] ),
        .O(rfmod_out[2]));
  LUT6 #(
    .INIT(64'hF0F6F0F000060000)) 
    \rfmod_out[3]_INST_0 
       (.I0(test_stat[23]),
        .I1(\u_tester/u_gen/out_reg_n_0_[2] ),
        .I2(\gpio_dir_reg_n_0_[3] ),
        .I3(test_dir),
        .I4(test_en),
        .I5(\gpio_dato_reg_n_0_[3] ),
        .O(rfmod_out[3]));
  LUT6 #(
    .INIT(64'hF0F6F0F000060000)) 
    \rfmod_out[4]_INST_0 
       (.I0(\u_tester/u_gen/out_reg_n_0_[3] ),
        .I1(test_stat[23]),
        .I2(\gpio_dir_reg_n_0_[4] ),
        .I3(test_dir),
        .I4(test_en),
        .I5(\gpio_dato_reg_n_0_[4] ),
        .O(rfmod_out[4]));
  LUT6 #(
    .INIT(64'hBBBB8B888B888B88)) 
    \rfmod_out[5]_INST_0 
       (.I0(\gpio_dato_reg_n_0_[5] ),
        .I1(p_2_in),
        .I2(\rfmod_out[9]_INST_0_i_1_n_0 ),
        .I3(rx_hisel),
        .I4(\rfmod_out[5]_INST_0_i_1_n_0 ),
        .I5(\rfmod_out[16]_INST_0_i_1_n_0 ),
        .O(rfmod_out[5]));
  (* SOFT_HLUTNM = "soft_lutpair160" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rfmod_out[5]_INST_0_i_1 
       (.I0(test_stat[23]),
        .I1(\u_tester/u_gen/out_reg_n_0_[0] ),
        .O(\rfmod_out[5]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hB888BBBBB8888888)) 
    \rfmod_out[6]_INST_0 
       (.I0(data0),
        .I1(\gpio_dir_reg_n_0_[6] ),
        .I2(\rfmod_out[6]_INST_0_i_1_n_0 ),
        .I3(test_dir),
        .I4(test_en),
        .I5(\rfmod_out[6]_INST_0_i_2_n_0 ),
        .O(rfmod_out[6]));
  (* SOFT_HLUTNM = "soft_lutpair160" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rfmod_out[6]_INST_0_i_1 
       (.I0(test_stat[23]),
        .I1(\u_tester/u_gen/out_reg_n_0_[1] ),
        .O(\rfmod_out[6]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h8000003C80000000)) 
    \rfmod_out[6]_INST_0_i_2 
       (.I0(tx_hisel),
        .I1(rfmod_id[1]),
        .I2(rfmod_id[0]),
        .I3(rfmod_id[2]),
        .I4(rfmod_id[3]),
        .I5(spi_mosi),
        .O(\rfmod_out[6]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAAAACF0FAAAA0000)) 
    \rfmod_out[7]_INST_0 
       (.I0(\gpio_dato_reg_n_0_[7] ),
        .I1(\rfmod_out[7]_INST_0_i_1_n_0 ),
        .I2(test_en),
        .I3(test_dir),
        .I4(\gpio_dir_reg_n_0_[7] ),
        .I5(\rfmod_out[7]_INST_0_i_2_n_0 ),
        .O(rfmod_out[7]));
  (* SOFT_HLUTNM = "soft_lutpair159" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rfmod_out[7]_INST_0_i_1 
       (.I0(test_stat[23]),
        .I1(\u_tester/u_gen/out_reg_n_0_[2] ),
        .O(\rfmod_out[7]_INST_0_i_1_n_0 ));
  LUT5 #(
    .INIT(32'hFFFF88F8)) 
    \rfmod_out[7]_INST_0_i_2 
       (.I0(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I1(\ctrl0_reg_n_0_[0] ),
        .I2(spi_cs_n),
        .I3(\rfmod_oe[16]_INST_0_i_1_n_0 ),
        .I4(test_en),
        .O(\rfmod_out[7]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hB888BBBBB8888888)) 
    \rfmod_out[8]_INST_0 
       (.I0(\gpio_dato_reg_n_0_[8] ),
        .I1(\gpio_dir_reg_n_0_[8] ),
        .I2(\rfmod_out[8]_INST_0_i_1_n_0 ),
        .I3(test_dir),
        .I4(test_en),
        .I5(\rfmod_out[8]_INST_0_i_2_n_0 ),
        .O(rfmod_out[8]));
  (* SOFT_HLUTNM = "soft_lutpair158" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rfmod_out[8]_INST_0_i_1 
       (.I0(\u_tester/u_gen/out_reg_n_0_[3] ),
        .I1(test_stat[23]),
        .O(\rfmod_out[8]_INST_0_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h8000000080000300)) 
    \rfmod_out[8]_INST_0_i_2 
       (.I0(fdd_en_b),
        .I1(rfmod_id[3]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[0]),
        .I4(rfmod_id[1]),
        .I5(\rfmod_out[9]_INST_0_i_2_n_0 ),
        .O(\rfmod_out[8]_INST_0_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hBBBB888B888B888B)) 
    \rfmod_out[9]_INST_0 
       (.I0(\gpio_dato_reg_n_0_[9] ),
        .I1(\gpio_dir_reg_n_0_[9] ),
        .I2(\rfmod_out[9]_INST_0_i_1_n_0 ),
        .I3(\rfmod_out[9]_INST_0_i_2_n_0 ),
        .I4(\rfmod_out[9]_INST_0_i_3_n_0 ),
        .I5(\rfmod_out[13]_INST_0_i_1_n_0 ),
        .O(rfmod_out[9]));
  (* SOFT_HLUTNM = "soft_lutpair60" *) 
  LUT5 #(
    .INIT(32'hBFFFFFFF)) 
    \rfmod_out[9]_INST_0_i_1 
       (.I0(test_en),
        .I1(rfmod_id[3]),
        .I2(rfmod_id[0]),
        .I3(rfmod_id[2]),
        .I4(rfmod_id[1]),
        .O(\rfmod_out[9]_INST_0_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair91" *) 
  LUT2 #(
    .INIT(4'h1)) 
    \rfmod_out[9]_INST_0_i_2 
       (.I0(tx_active[0]),
        .I1(p_0_in0_in),
        .O(\rfmod_out[9]_INST_0_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair157" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \rfmod_out[9]_INST_0_i_3 
       (.I0(test_en),
        .I1(test_dir),
        .O(\rfmod_out[9]_INST_0_i_3_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \rx_err_cnt[0]_i_1 
       (.I0(test_stat__0[0]),
        .O(p_0_in__2[0]));
  (* SOFT_HLUTNM = "soft_lutpair107" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rx_err_cnt[1]_i_1 
       (.I0(test_stat__0[0]),
        .I1(test_stat__0[1]),
        .O(p_0_in__2[1]));
  (* SOFT_HLUTNM = "soft_lutpair107" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \rx_err_cnt[2]_i_1 
       (.I0(test_stat__0[2]),
        .I1(test_stat__0[1]),
        .I2(test_stat__0[0]),
        .O(p_0_in__2[2]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \rx_err_cnt[3]_i_1 
       (.I0(test_stat__0[3]),
        .I1(test_stat__0[0]),
        .I2(test_stat__0[1]),
        .I3(test_stat__0[2]),
        .O(p_0_in__2[3]));
  (* SOFT_HLUTNM = "soft_lutpair15" *) 
  LUT5 #(
    .INIT(32'h6AAAAAAA)) 
    \rx_err_cnt[4]_i_1 
       (.I0(test_stat__0[4]),
        .I1(test_stat__0[2]),
        .I2(test_stat__0[1]),
        .I3(test_stat__0[0]),
        .I4(test_stat__0[3]),
        .O(p_0_in__2[4]));
  LUT6 #(
    .INIT(64'h6AAAAAAAAAAAAAAA)) 
    \rx_err_cnt[5]_i_1 
       (.I0(test_stat__0[5]),
        .I1(test_stat__0[3]),
        .I2(test_stat__0[0]),
        .I3(test_stat__0[1]),
        .I4(test_stat__0[2]),
        .I5(test_stat__0[4]),
        .O(p_0_in__2[5]));
  (* SOFT_HLUTNM = "soft_lutpair105" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \rx_err_cnt[6]_i_1 
       (.I0(test_stat__0[6]),
        .I1(\rx_err_cnt[7]_i_3_n_0 ),
        .O(p_0_in__2[6]));
  LUT6 #(
    .INIT(64'h7F007F007F000000)) 
    \rx_err_cnt[7]_i_1 
       (.I0(test_stat__0[6]),
        .I1(\rx_err_cnt[7]_i_3_n_0 ),
        .I2(test_stat__0[7]),
        .I3(test_stat[21]),
        .I4(\u_tester/st_cur [1]),
        .I5(\u_tester/st_cur [0]),
        .O(\u_tester/rx_err_cnt0 ));
  (* SOFT_HLUTNM = "soft_lutpair105" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \rx_err_cnt[7]_i_2 
       (.I0(test_stat__0[7]),
        .I1(\rx_err_cnt[7]_i_3_n_0 ),
        .I2(test_stat__0[6]),
        .O(p_0_in__2[7]));
  LUT6 #(
    .INIT(64'h8000000000000000)) 
    \rx_err_cnt[7]_i_3 
       (.I0(test_stat__0[5]),
        .I1(test_stat__0[3]),
        .I2(test_stat__0[0]),
        .I3(test_stat__0[1]),
        .I4(test_stat__0[2]),
        .I5(test_stat__0[4]),
        .O(\rx_err_cnt[7]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h1110000011101110)) 
    rx_err_s_i_1
       (.I0(rst),
        .I1(test_stat[22]),
        .I2(\u_tester/st_cur [1]),
        .I3(\u_tester/st_cur [0]),
        .I4(test_stat[21]),
        .I5(\st_cur[0]_i_2_n_0 ),
        .O(rx_err_s_i_1_n_0));
  LUT6 #(
    .INIT(64'hF7FF7FF700882022)) 
    scl_o_reg_i_1
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_next ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .I5(scl_t),
        .O(scl_o_reg_i_1_n_0));
  LUT3 #(
    .INIT(8'h08)) 
    \scratch[15]_i_1 
       (.I0(en),
        .I1(wr),
        .I2(\scratch[15]_i_2_n_0 ),
        .O(\scratch[15]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h7FFFFFFFFFFFFFFF)) 
    \scratch[15]_i_2 
       (.I0(addr[0]),
        .I1(addr[4]),
        .I2(addr[3]),
        .I3(addr[1]),
        .I4(addr[5]),
        .I5(addr[2]),
        .O(\scratch[15]_i_2_n_0 ));
  FDSE \scratch_reg[0] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[0]),
        .Q(scratch[0]),
        .S(rst));
  FDSE \scratch_reg[10] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[10]),
        .Q(scratch[10]),
        .S(rst));
  FDSE \scratch_reg[11] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[11]),
        .Q(scratch[11]),
        .S(rst));
  FDSE \scratch_reg[12] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[12]),
        .Q(scratch[12]),
        .S(rst));
  FDSE \scratch_reg[13] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[13]),
        .Q(scratch[13]),
        .S(rst));
  FDRE \scratch_reg[14] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[14]),
        .Q(scratch[14]),
        .R(rst));
  FDRE \scratch_reg[15] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[15]),
        .Q(scratch[15]),
        .R(rst));
  FDSE \scratch_reg[1] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[1]),
        .Q(scratch[1]),
        .S(rst));
  FDSE \scratch_reg[2] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[2]),
        .Q(scratch[2]),
        .S(rst));
  FDSE \scratch_reg[3] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[3]),
        .Q(scratch[3]),
        .S(rst));
  FDRE \scratch_reg[4] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[4]),
        .Q(scratch[4]),
        .R(rst));
  FDSE \scratch_reg[5] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[5]),
        .Q(scratch[5]),
        .S(rst));
  FDSE \scratch_reg[6] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[6]),
        .Q(scratch[6]),
        .S(rst));
  FDSE \scratch_reg[7] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[7]),
        .Q(scratch[7]),
        .S(rst));
  FDRE \scratch_reg[8] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[8]),
        .Q(scratch[8]),
        .R(rst));
  FDSE \scratch_reg[9] 
       (.C(clk),
        .CE(\scratch[15]_i_1_n_0 ),
        .D(dati[9]),
        .Q(scratch[9]),
        .S(rst));
  LUT6 #(
    .INIT(64'hFFFFB8FF0000B800)) 
    sda_o_reg_i_1
       (.I0(sda_o_reg_i_2_n_0),
        .I1(sda_o_reg_i_3_n_0),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_next ),
        .I4(sda_o_reg_i_4_n_0),
        .I5(sda_t),
        .O(sda_o_reg_i_1_n_0));
  LUT5 #(
    .INIT(32'h2A222222)) 
    sda_o_reg_i_10
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .I1(sda_o_reg_i_14_n_0),
        .I2(sda_o_reg_i_15_n_0),
        .I3(\FSM_onehot_state_reg[9]_i_2_n_0 ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/mode_read_reg ),
        .O(sda_o_reg_i_10_n_0));
  LUT6 #(
    .INIT(64'hFACF0ACFFAC00AC0)) 
    sda_o_reg_i_11
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [1]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [2]),
        .I2(\u_i2c_master/bit_count_reg [1]),
        .I3(\u_i2c_master/bit_count_reg [0]),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [3]),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [4]),
        .O(sda_o_reg_i_11_n_0));
  (* SOFT_HLUTNM = "soft_lutpair56" *) 
  LUT2 #(
    .INIT(4'h1)) 
    sda_o_reg_i_12
       (.I0(\u_i2c_master/bit_count_reg [1]),
        .I1(\u_i2c_master/bit_count_reg [0]),
        .O(sda_o_reg_i_12_n_0));
  LUT6 #(
    .INIT(64'hFACF0ACFFAC00AC0)) 
    sda_o_reg_i_13
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [5]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [6]),
        .I2(\u_i2c_master/bit_count_reg [1]),
        .I3(\u_i2c_master/bit_count_reg [0]),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [7]),
        .I5(\u_i2c_master/u_i2c_master/u_i2c_master/data_reg_reg_n_0_[7] ),
        .O(sda_o_reg_i_13_n_0));
  LUT5 #(
    .INIT(32'h30073FF7)) 
    sda_o_reg_i_14
       (.I0(\u_i2c_master/bit_count_reg [3]),
        .I1(sda_o_reg_i_16_n_0),
        .I2(\u_i2c_master/bit_count_reg [2]),
        .I3(\u_i2c_master/bit_count_reg [1]),
        .I4(sda_o_reg_i_17_n_0),
        .O(sda_o_reg_i_14_n_0));
  (* SOFT_HLUTNM = "soft_lutpair28" *) 
  LUT3 #(
    .INIT(8'hFE)) 
    sda_o_reg_i_15
       (.I0(\u_i2c_master/bit_count_reg [1]),
        .I1(\u_i2c_master/bit_count_reg [3]),
        .I2(\u_i2c_master/bit_count_reg [2]),
        .O(sda_o_reg_i_15_n_0));
  LUT5 #(
    .INIT(32'hB8FFB800)) 
    sda_o_reg_i_16
       (.I0(\u_i2c_master/addr_reg [5]),
        .I1(\u_i2c_master/bit_count_reg [0]),
        .I2(\u_i2c_master/addr_reg [4]),
        .I3(\u_i2c_master/bit_count_reg [1]),
        .I4(\u_i2c_master/addr_reg [6]),
        .O(sda_o_reg_i_16_n_0));
  LUT6 #(
    .INIT(64'hAFA0CFCFAFA0C0C0)) 
    sda_o_reg_i_17
       (.I0(\u_i2c_master/addr_reg [1]),
        .I1(\u_i2c_master/addr_reg [0]),
        .I2(\u_i2c_master/bit_count_reg [1]),
        .I3(\u_i2c_master/addr_reg [3]),
        .I4(\u_i2c_master/bit_count_reg [0]),
        .I5(\u_i2c_master/addr_reg [2]),
        .O(sda_o_reg_i_17_n_0));
  LUT6 #(
    .INIT(64'hFFFF0DFF0DFF0DFF)) 
    sda_o_reg_i_2
       (.I0(sda_o_reg_i_5_n_0),
        .I1(\FSM_onehot_state_reg[6]_i_1_n_0 ),
        .I2(\addr_reg[6]_i_2_n_0 ),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I4(\FSM_sequential_phy_state_reg[3]_i_10_n_0 ),
        .I5(sda_o_reg_i_6_n_0),
        .O(sda_o_reg_i_2_n_0));
  (* SOFT_HLUTNM = "soft_lutpair70" *) 
  LUT2 #(
    .INIT(4'h1)) 
    sda_o_reg_i_3
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I1(\FSM_sequential_phy_state_reg[0]_i_3_n_0 ),
        .O(sda_o_reg_i_3_n_0));
  LUT6 #(
    .INIT(64'h00000000AAAAAAAE)) 
    sda_o_reg_i_4
       (.I0(sda_o_reg_i_7_n_0),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I2(\FSM_sequential_phy_state_reg[0]_i_3_n_0 ),
        .I3(\FSM_sequential_phy_state_reg[0]_i_2_n_0 ),
        .I4(\FSM_sequential_phy_state_reg[3]_i_4_n_0 ),
        .I5(sda_o_reg_i_8_n_0),
        .O(sda_o_reg_i_4_n_0));
  LUT6 #(
    .INIT(64'h000000000000DD0D)) 
    sda_o_reg_i_5
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg ),
        .I1(\FSM_sequential_phy_state_reg[0]_i_4_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I3(\FSM_onehot_state_reg[7]_i_3_n_0 ),
        .I4(sda_o_reg_i_9_n_0),
        .I5(sda_o_reg_i_10_n_0),
        .O(sda_o_reg_i_5_n_0));
  LUT6 #(
    .INIT(64'hAA2AAA2A0000AA2A)) 
    sda_o_reg_i_6
       (.I0(\FSM_sequential_phy_state_reg[0]_i_5_n_0 ),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg ),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .I3(\FSM_onehot_state_reg[9]_i_2_n_0 ),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .I5(\FSM_onehot_state_reg[7]_i_3_n_0 ),
        .O(sda_o_reg_i_6_n_0));
  (* SOFT_HLUTNM = "soft_lutpair14" *) 
  LUT4 #(
    .INIT(16'hFF1F)) 
    sda_o_reg_i_7
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .O(sda_o_reg_i_7_n_0));
  (* SOFT_HLUTNM = "soft_lutpair53" *) 
  LUT4 #(
    .INIT(16'h2001)) 
    sda_o_reg_i_8
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .I1(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .I2(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .I3(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .O(sda_o_reg_i_8_n_0));
  LUT6 #(
    .INIT(64'h8AA8800880A88008)) 
    sda_o_reg_i_9
       (.I0(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .I1(sda_o_reg_i_11_n_0),
        .I2(\u_i2c_master/bit_count_reg [2]),
        .I3(sda_o_reg_i_12_n_0),
        .I4(sda_o_reg_i_13_n_0),
        .I5(\u_i2c_master/bit_count_reg [3]),
        .O(sda_o_reg_i_9_n_0));
  (* SOFT_HLUTNM = "soft_lutpair184" *) 
  LUT1 #(
    .INIT(2'h1)) 
    \slip_cnt[0]_i_1 
       (.I0(test_stat__0[16]),
        .O(p_0_in__3[0]));
  (* SOFT_HLUTNM = "soft_lutpair184" *) 
  LUT2 #(
    .INIT(4'h6)) 
    \slip_cnt[1]_i_1 
       (.I0(test_stat__0[16]),
        .I1(test_stat__0[17]),
        .O(p_0_in__3[1]));
  (* SOFT_HLUTNM = "soft_lutpair98" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \slip_cnt[2]_i_1 
       (.I0(test_stat__0[18]),
        .I1(test_stat__0[17]),
        .I2(test_stat__0[16]),
        .O(p_0_in__3[2]));
  LUT2 #(
    .INIT(4'hB)) 
    \slip_cnt[3]_i_1 
       (.I0(rst),
        .I1(test_en),
        .O(\u_tester/slip_cnt0 ));
  LUT2 #(
    .INIT(4'h1)) 
    \slip_cnt[3]_i_2 
       (.I0(\u_tester/st_cur [0]),
        .I1(\u_tester/st_cur [1]),
        .O(\slip_cnt[3]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair98" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \slip_cnt[3]_i_3 
       (.I0(test_stat__0[19]),
        .I1(test_stat__0[16]),
        .I2(test_stat__0[17]),
        .I3(test_stat__0[18]),
        .O(p_0_in__3[3]));
  (* SOFT_HLUTNM = "soft_lutpair144" *) 
  LUT2 #(
    .INIT(4'h2)) 
    \spi_bit_cnt[0]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[0] ),
        .O(\spi_bit_cnt[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair88" *) 
  LUT3 #(
    .INIT(8'h60)) 
    \spi_bit_cnt[1]_i_1 
       (.I0(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[1] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[0] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .O(\spi_bit_cnt[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair48" *) 
  LUT4 #(
    .INIT(16'h2A80)) 
    \spi_bit_cnt[2]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[0] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[1] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[2] ),
        .O(\spi_bit_cnt[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair48" *) 
  LUT5 #(
    .INIT(32'h2AAA8000)) 
    \spi_bit_cnt[3]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[1] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[0] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[2] ),
        .I4(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[3] ),
        .O(\spi_bit_cnt[3]_i_1_n_0 ));
  LUT3 #(
    .INIT(8'hEA)) 
    \spi_bit_cnt[4]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\FSM_onehot_state[5]_i_3_n_0 ),
        .O(\u_spi_gpio_exp_master/r_inbus ));
  LUT6 #(
    .INIT(64'h7FFF800000000000)) 
    \spi_bit_cnt[4]_i_2 
       (.I0(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[0] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[1] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[3] ),
        .I4(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[4] ),
        .I5(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .O(\spi_bit_cnt[4]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair41" *) 
  LUT5 #(
    .INIT(32'h0000FFFE)) 
    \spi_bit_div[0]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[3] ),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[1] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I4(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[0] ),
        .O(\spi_bit_div[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000FFFEFFFE0000)) 
    \spi_bit_div[1]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[3] ),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[1] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I4(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[0] ),
        .I5(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[1] ),
        .O(\spi_bit_div[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair5" *) 
  LUT5 #(
    .INIT(32'h04444000)) 
    \spi_bit_div[2]_i_1 
       (.I0(\FSM_onehot_state[5]_i_3_n_0 ),
        .I1(\spi_bit_div[6]_i_3_n_0 ),
        .I2(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[1] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[0] ),
        .I4(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[2] ),
        .O(\spi_bit_div[2]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h2AAA8000)) 
    \spi_bit_div[3]_i_1 
       (.I0(\spi_bit_div[6]_i_3_n_0 ),
        .I1(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[0] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[1] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[2] ),
        .I4(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[3] ),
        .O(\spi_bit_div[3]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h2AAAAAAA80000000)) 
    \spi_bit_div[4]_i_1 
       (.I0(\spi_bit_div[6]_i_3_n_0 ),
        .I1(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[1] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[0] ),
        .I4(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[3] ),
        .I5(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[4] ),
        .O(\spi_bit_div[4]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'h0220)) 
    \spi_bit_div[5]_i_1 
       (.I0(\spi_bit_div[6]_i_3_n_0 ),
        .I1(\FSM_onehot_state[5]_i_3_n_0 ),
        .I2(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[5] ),
        .I3(\spi_bit_div[5]_i_2_n_0 ),
        .O(\spi_bit_div[5]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h80000000)) 
    \spi_bit_div[5]_i_2 
       (.I0(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[3] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[0] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[1] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[2] ),
        .I4(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[4] ),
        .O(\spi_bit_div[5]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \spi_bit_div[6]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[3] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[1] ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .I4(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .O(\u_spi_gpio_exp_master/spi_bit_div ));
  LUT4 #(
    .INIT(16'h0028)) 
    \spi_bit_div[6]_i_2 
       (.I0(\spi_bit_div[6]_i_3_n_0 ),
        .I1(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[6] ),
        .I2(\spi_bit_div[6]_i_4_n_0 ),
        .I3(\FSM_onehot_state[5]_i_3_n_0 ),
        .O(\spi_bit_div[6]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair41" *) 
  LUT4 #(
    .INIT(16'hFFFE)) 
    \spi_bit_div[6]_i_3 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[1] ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[3] ),
        .O(\spi_bit_div[6]_i_3_n_0 ));
  LUT6 #(
    .INIT(64'h8000000000000000)) 
    \spi_bit_div[6]_i_4 
       (.I0(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[4] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[1] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[0] ),
        .I4(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[3] ),
        .I5(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[5] ),
        .O(\spi_bit_div[6]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair96" *) 
  LUT4 #(
    .INIT(16'hFBFA)) 
    spi_cs_n_i_1
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[1] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .I3(spi_cs_n),
        .O(spi_cs_n_i_1_n_0));
  LUT5 #(
    .INIT(32'h8F888088)) 
    spi_mosi_i_1
       (.I0(\u_spi_gpio_exp_master/p_0_in ),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/w_outbus ),
        .I3(spi_mosi_i_2_n_0),
        .I4(spi_mosi),
        .O(spi_mosi_i_1_n_0));
  (* SOFT_HLUTNM = "soft_lutpair96" *) 
  LUT3 #(
    .INIT(8'h01)) 
    spi_mosi_i_2
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[3] ),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[1] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .O(spi_mosi_i_2_n_0));
  LUT6 #(
    .INIT(64'hF0F2F2F2F0F0F0F0)) 
    spi_sclk_i_1
       (.I0(spi_mosi_i_2_n_0),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I2(spi_sclk_i_2_n_0),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I4(\FSM_onehot_state[5]_i_3_n_0 ),
        .I5(spi_sclk),
        .O(spi_sclk_i_1_n_0));
  LUT5 #(
    .INIT(32'h00200000)) 
    spi_sclk_i_2
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I1(\w_outbus[31]_i_3_n_0 ),
        .I2(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[4] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[1] ),
        .I4(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[5] ),
        .O(spi_sclk_i_2_n_0));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT4 #(
    .INIT(16'hBF40)) 
    \st_addr[0]_i_1 
       (.I0(\u_i2c_master/state_reg_n_0_[1] ),
        .I1(\u_i2c_master/state_reg_n_0_[0] ),
        .I2(\u_i2c_master/state_reg_n_0_[2] ),
        .I3(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .O(\st_addr[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair31" *) 
  LUT5 #(
    .INIT(32'hFF7F0080)) 
    \st_addr[1]_i_1 
       (.I0(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .I1(\u_i2c_master/state_reg_n_0_[2] ),
        .I2(\u_i2c_master/state_reg_n_0_[0] ),
        .I3(\u_i2c_master/state_reg_n_0_[1] ),
        .I4(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .O(\st_addr[1]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFF7FFF00008000)) 
    \st_addr[2]_i_1 
       (.I0(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .I1(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .I2(\u_i2c_master/state_reg_n_0_[2] ),
        .I3(\u_i2c_master/state_reg_n_0_[0] ),
        .I4(\u_i2c_master/state_reg_n_0_[1] ),
        .I5(\u_i2c_master/st_addr_reg_n_0_[2] ),
        .O(\st_addr[2]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0E0E0E0E0E000E0E)) 
    \st_cur[0]_i_1 
       (.I0(\st_cur[0]_i_2_n_0 ),
        .I1(\u_tester/st_cur [0]),
        .I2(\st_cur[0]_i_3_n_0 ),
        .I3(\st_cur[0]_i_4_n_0 ),
        .I4(test_stat[21]),
        .I5(\u_tester/st_cur [1]),
        .O(\st_cur[0]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair22" *) 
  LUT4 #(
    .INIT(16'h56A6)) 
    \st_cur[0]_i_10 
       (.I0(\u_tester/chk_out [1]),
        .I1(rfmod_in[6]),
        .I2(test_dir),
        .I3(rfmod_in[2]),
        .O(\st_cur[0]_i_10_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair27" *) 
  LUT4 #(
    .INIT(16'h56A6)) 
    \st_cur[0]_i_11 
       (.I0(\u_tester/chk_out [5]),
        .I1(rfmod_in[14]),
        .I2(test_dir),
        .I3(rfmod_in[10]),
        .O(\st_cur[0]_i_11_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair21" *) 
  LUT4 #(
    .INIT(16'h56A6)) 
    \st_cur[0]_i_12 
       (.I0(\u_tester/chk_out [6]),
        .I1(rfmod_in[15]),
        .I2(test_dir),
        .I3(rfmod_in[11]),
        .O(\st_cur[0]_i_12_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair29" *) 
  LUT4 #(
    .INIT(16'h56A6)) 
    \st_cur[0]_i_13 
       (.I0(\u_tester/chk_out [0]),
        .I1(rfmod_in[5]),
        .I2(test_dir),
        .I3(rfmod_in[1]),
        .O(\st_cur[0]_i_13_n_0 ));
  LUT5 #(
    .INIT(32'h00000001)) 
    \st_cur[0]_i_2 
       (.I0(\st_cur[0]_i_5_n_0 ),
        .I1(\st_cur[0]_i_6_n_0 ),
        .I2(\st_cur[0]_i_7_n_0 ),
        .I3(\st_cur[0]_i_8_n_0 ),
        .I4(\st_cur[0]_i_9_n_0 ),
        .O(\st_cur[0]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFFFABFFFF)) 
    \st_cur[0]_i_3 
       (.I0(\st_cur[1]_i_3_n_0 ),
        .I1(\u_tester/st_cur [1]),
        .I2(\u_tester/st_cur [0]),
        .I3(test_stat[29]),
        .I4(test_en),
        .I5(rst),
        .O(\st_cur[0]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair173" *) 
  LUT2 #(
    .INIT(4'h7)) 
    \st_cur[0]_i_4 
       (.I0(\u_tester/count_reg [6]),
        .I1(\count[6]_i_2_n_0 ),
        .O(\st_cur[0]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair18" *) 
  LUT4 #(
    .INIT(16'h56A6)) 
    \st_cur[0]_i_5 
       (.I0(\u_tester/chk_out [7]),
        .I1(rfmod_in[16]),
        .I2(test_dir),
        .I3(rfmod_in[12]),
        .O(\st_cur[0]_i_5_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair23" *) 
  LUT4 #(
    .INIT(16'h56A6)) 
    \st_cur[0]_i_6 
       (.I0(\u_tester/chk_out [2]),
        .I1(rfmod_in[7]),
        .I2(test_dir),
        .I3(rfmod_in[3]),
        .O(\st_cur[0]_i_6_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair16" *) 
  LUT4 #(
    .INIT(16'h56A6)) 
    \st_cur[0]_i_7 
       (.I0(\u_tester/chk_out [3]),
        .I1(rfmod_in[8]),
        .I2(test_dir),
        .I3(rfmod_in[4]),
        .O(\st_cur[0]_i_7_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair17" *) 
  LUT4 #(
    .INIT(16'h56A6)) 
    \st_cur[0]_i_8 
       (.I0(\u_tester/chk_out [4]),
        .I1(rfmod_in[13]),
        .I2(test_dir),
        .I3(rfmod_in[9]),
        .O(\st_cur[0]_i_8_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \st_cur[0]_i_9 
       (.I0(\st_cur[0]_i_10_n_0 ),
        .I1(\st_cur[0]_i_11_n_0 ),
        .I2(\st_cur[0]_i_12_n_0 ),
        .I3(\st_cur[0]_i_13_n_0 ),
        .O(\st_cur[0]_i_9_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair97" *) 
  LUT4 #(
    .INIT(16'h0020)) 
    \st_cur[1]_i_1 
       (.I0(\st_cur[1]_i_2_n_0 ),
        .I1(rst),
        .I2(test_en),
        .I3(test_stat[29]),
        .O(\st_cur[1]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hFFFFFFFFF01FFA1F)) 
    \st_cur[1]_i_2 
       (.I0(\st_cur[0]_i_4_n_0 ),
        .I1(test_stat[21]),
        .I2(\u_tester/st_cur [0]),
        .I3(\u_tester/st_cur [1]),
        .I4(\st_cur[0]_i_2_n_0 ),
        .I5(\st_cur[1]_i_3_n_0 ),
        .O(\st_cur[1]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'h00000001)) 
    \st_cur[1]_i_3 
       (.I0(\u_tester/mask_count_reg_n_0_[7] ),
        .I1(\u_tester/mask_count_reg_n_0_[5] ),
        .I2(\u_tester/mask_count_reg_n_0_[2] ),
        .I3(\u_tester/mask_count_reg_n_0_[0] ),
        .I4(\st_cur[1]_i_4_n_0 ),
        .O(\st_cur[1]_i_3_n_0 ));
  LUT4 #(
    .INIT(16'hFFFE)) 
    \st_cur[1]_i_4 
       (.I0(\u_tester/mask_count_reg_n_0_[4] ),
        .I1(\u_tester/mask_count_reg_n_0_[6] ),
        .I2(\u_tester/mask_count_reg_n_0_[1] ),
        .I3(\u_tester/mask_count_reg_n_0_[3] ),
        .O(\st_cur[1]_i_4_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT5 #(
    .INIT(32'h00FF1F00)) 
    \state[0]_i_1 
       (.I0(\u_i2c_master/state_reg_n_0_[1] ),
        .I1(\state[1]_i_2_n_0 ),
        .I2(\u_i2c_master/state_reg_n_0_[2] ),
        .I3(\state[2]_i_3_n_0 ),
        .I4(\u_i2c_master/state_reg_n_0_[0] ),
        .O(\state[0]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'h0000FFFF37340000)) 
    \state[1]_i_1 
       (.I0(\state[1]_i_2_n_0 ),
        .I1(\u_i2c_master/state_reg_n_0_[2] ),
        .I2(\u_i2c_master/state_reg_n_0_[0] ),
        .I3(i2c_busy),
        .I4(\state[2]_i_3_n_0 ),
        .I5(\u_i2c_master/state_reg_n_0_[1] ),
        .O(\state[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair24" *) 
  LUT5 #(
    .INIT(32'h00000080)) 
    \state[1]_i_2 
       (.I0(\u_i2c_master/u_i2c_master/state [0]),
        .I1(\u_i2c_master/u_i2c_master/state [1]),
        .I2(\u_i2c_master/u_i2c_master/state [2]),
        .I3(\u_i2c_master/u_i2c_master/state [3]),
        .I4(\u_i2c_master/u_i2c_master/u_i2c_master/busy_reg_reg_n_0 ),
        .O(\state[1]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAFFFFAAAB0000)) 
    \state[2]_i_1 
       (.I0(\state[2]_i_2_n_0 ),
        .I1(\i2c_addr_s[6]_i_3_n_0 ),
        .I2(i2c_busy),
        .I3(\u_i2c_master/state_reg_n_0_[0] ),
        .I4(\state[2]_i_3_n_0 ),
        .I5(\u_i2c_master/state_reg_n_0_[2] ),
        .O(\state[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair0" *) 
  LUT4 #(
    .INIT(16'h0C1C)) 
    \state[2]_i_2 
       (.I0(\state[1]_i_2_n_0 ),
        .I1(\u_i2c_master/state_reg_n_0_[1] ),
        .I2(\u_i2c_master/state_reg_n_0_[2] ),
        .I3(\u_i2c_master/state_reg_n_0_[0] ),
        .O(\state[2]_i_2_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair12" *) 
  LUT5 #(
    .INIT(32'hFFFC32CE)) 
    \state[2]_i_3 
       (.I0(\state[2]_i_4_n_0 ),
        .I1(\u_i2c_master/state_reg_n_0_[0] ),
        .I2(\u_i2c_master/state_reg_n_0_[1] ),
        .I3(\state[1]_i_2_n_0 ),
        .I4(\u_i2c_master/state_reg_n_0_[2] ),
        .O(\state[2]_i_3_n_0 ));
  LUT5 #(
    .INIT(32'hFFFFFFFE)) 
    \state[2]_i_4 
       (.I0(i2c_busy),
        .I1(\u_i2c_master/prog_cache_entries [0]),
        .I2(\u_i2c_master/prog_cache_entries [1]),
        .I3(\u_i2c_master/prog_cache_entries [2]),
        .I4(\u_i2c_master/prog_cache_entries [3]),
        .O(\state[2]_i_4_n_0 ));
  LUT6 #(
    .INIT(64'h0010000000000000)) 
    \test_ctrl[15]_i_1 
       (.I0(addr[2]),
        .I1(addr[5]),
        .I2(addr[3]),
        .I3(addr[4]),
        .I4(addr[1]),
        .I5(\test_ctrl[15]_i_2_n_0 ),
        .O(__do_out193_out));
  (* SOFT_HLUTNM = "soft_lutpair37" *) 
  LUT3 #(
    .INIT(8'h80)) 
    \test_ctrl[15]_i_2 
       (.I0(en),
        .I1(wr),
        .I2(addr[0]),
        .O(\test_ctrl[15]_i_2_n_0 ));
  FDRE #(
    .INIT(1'b0)) 
    \test_ctrl_reg[11] 
       (.C(clk),
        .CE(__do_out193_out),
        .D(dati[11]),
        .Q(test_dir),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \test_ctrl_reg[12] 
       (.C(clk),
        .CE(__do_out193_out),
        .D(dati[12]),
        .Q(test_en),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \test_ctrl_reg[13] 
       (.C(clk),
        .CE(__do_out193_out),
        .D(dati[13]),
        .Q(test_stat[23]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \test_ctrl_reg[14] 
       (.C(clk),
        .CE(__do_out193_out),
        .D(dati[14]),
        .Q(test_stat[22]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \test_ctrl_reg[15] 
       (.C(clk),
        .CE(__do_out193_out),
        .D(dati[15]),
        .Q(test_stat[29]),
        .R(rst));
  LUT3 #(
    .INIT(8'h80)) 
    \tx_cnt[0]_i_1 
       (.I0(\u_tester/st_cur [0]),
        .I1(\u_tester/st_cur [1]),
        .I2(test_en),
        .O(\tx_cnt[0]_i_1_n_0 ));
  LUT1 #(
    .INIT(2'h1)) 
    \tx_cnt[0]_i_3 
       (.I0(\u_tester/tx_cnt_reg_n_0_[0] ),
        .O(\tx_cnt[0]_i_3_n_0 ));
  CARRY4 \tx_cnt_reg[0]_i_2 
       (.CI(1'b0),
        .CO({\tx_cnt_reg[0]_i_2_n_0 ,\tx_cnt_reg[0]_i_2_n_1 ,\tx_cnt_reg[0]_i_2_n_2 ,\tx_cnt_reg[0]_i_2_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b1}),
        .O({\tx_cnt_reg[0]_i_2_n_4 ,\tx_cnt_reg[0]_i_2_n_5 ,\tx_cnt_reg[0]_i_2_n_6 ,\tx_cnt_reg[0]_i_2_n_7 }),
        .S({\u_tester/tx_cnt_reg_n_0_[3] ,\u_tester/tx_cnt_reg_n_0_[2] ,\u_tester/tx_cnt_reg_n_0_[1] ,\tx_cnt[0]_i_3_n_0 }));
  CARRY4 \tx_cnt_reg[12]_i_1 
       (.CI(\tx_cnt_reg[8]_i_1_n_0 ),
        .CO({\tx_cnt_reg[12]_i_1_n_0 ,\tx_cnt_reg[12]_i_1_n_1 ,\tx_cnt_reg[12]_i_1_n_2 ,\tx_cnt_reg[12]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[12]_i_1_n_4 ,\tx_cnt_reg[12]_i_1_n_5 ,\tx_cnt_reg[12]_i_1_n_6 ,\tx_cnt_reg[12]_i_1_n_7 }),
        .S({\u_tester/tx_cnt_reg_n_0_[15] ,\u_tester/tx_cnt_reg_n_0_[14] ,\u_tester/tx_cnt_reg_n_0_[13] ,\u_tester/tx_cnt_reg_n_0_[12] }));
  CARRY4 \tx_cnt_reg[16]_i_1 
       (.CI(\tx_cnt_reg[12]_i_1_n_0 ),
        .CO({\tx_cnt_reg[16]_i_1_n_0 ,\tx_cnt_reg[16]_i_1_n_1 ,\tx_cnt_reg[16]_i_1_n_2 ,\tx_cnt_reg[16]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[16]_i_1_n_4 ,\tx_cnt_reg[16]_i_1_n_5 ,\tx_cnt_reg[16]_i_1_n_6 ,\tx_cnt_reg[16]_i_1_n_7 }),
        .S({\u_tester/tx_cnt_reg_n_0_[19] ,\u_tester/tx_cnt_reg_n_0_[18] ,\u_tester/tx_cnt_reg_n_0_[17] ,\u_tester/tx_cnt_reg_n_0_[16] }));
  CARRY4 \tx_cnt_reg[20]_i_1 
       (.CI(\tx_cnt_reg[16]_i_1_n_0 ),
        .CO({\tx_cnt_reg[20]_i_1_n_0 ,\tx_cnt_reg[20]_i_1_n_1 ,\tx_cnt_reg[20]_i_1_n_2 ,\tx_cnt_reg[20]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[20]_i_1_n_4 ,\tx_cnt_reg[20]_i_1_n_5 ,\tx_cnt_reg[20]_i_1_n_6 ,\tx_cnt_reg[20]_i_1_n_7 }),
        .S({test_stat__0[8],\u_tester/tx_cnt_reg_n_0_[22] ,\u_tester/tx_cnt_reg_n_0_[21] ,\u_tester/tx_cnt_reg_n_0_[20] }));
  CARRY4 \tx_cnt_reg[24]_i_1 
       (.CI(\tx_cnt_reg[20]_i_1_n_0 ),
        .CO({\tx_cnt_reg[24]_i_1_n_0 ,\tx_cnt_reg[24]_i_1_n_1 ,\tx_cnt_reg[24]_i_1_n_2 ,\tx_cnt_reg[24]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[24]_i_1_n_4 ,\tx_cnt_reg[24]_i_1_n_5 ,\tx_cnt_reg[24]_i_1_n_6 ,\tx_cnt_reg[24]_i_1_n_7 }),
        .S(test_stat__0[12:9]));
  CARRY4 \tx_cnt_reg[28]_i_1 
       (.CI(\tx_cnt_reg[24]_i_1_n_0 ),
        .CO({\NLW_tx_cnt_reg[28]_i_1_CO_UNCONNECTED [3:2],\tx_cnt_reg[28]_i_1_n_2 ,\tx_cnt_reg[28]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\NLW_tx_cnt_reg[28]_i_1_O_UNCONNECTED [3],\tx_cnt_reg[28]_i_1_n_5 ,\tx_cnt_reg[28]_i_1_n_6 ,\tx_cnt_reg[28]_i_1_n_7 }),
        .S({1'b0,test_stat__0[15:13]}));
  CARRY4 \tx_cnt_reg[4]_i_1 
       (.CI(\tx_cnt_reg[0]_i_2_n_0 ),
        .CO({\tx_cnt_reg[4]_i_1_n_0 ,\tx_cnt_reg[4]_i_1_n_1 ,\tx_cnt_reg[4]_i_1_n_2 ,\tx_cnt_reg[4]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[4]_i_1_n_4 ,\tx_cnt_reg[4]_i_1_n_5 ,\tx_cnt_reg[4]_i_1_n_6 ,\tx_cnt_reg[4]_i_1_n_7 }),
        .S({\u_tester/tx_cnt_reg_n_0_[7] ,\u_tester/tx_cnt_reg_n_0_[6] ,\u_tester/tx_cnt_reg_n_0_[5] ,\u_tester/tx_cnt_reg_n_0_[4] }));
  CARRY4 \tx_cnt_reg[8]_i_1 
       (.CI(\tx_cnt_reg[4]_i_1_n_0 ),
        .CO({\tx_cnt_reg[8]_i_1_n_0 ,\tx_cnt_reg[8]_i_1_n_1 ,\tx_cnt_reg[8]_i_1_n_2 ,\tx_cnt_reg[8]_i_1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b0,1'b0,1'b0}),
        .O({\tx_cnt_reg[8]_i_1_n_4 ,\tx_cnt_reg[8]_i_1_n_5 ,\tx_cnt_reg[8]_i_1_n_6 ,\tx_cnt_reg[8]_i_1_n_7 }),
        .S({\u_tester/tx_cnt_reg_n_0_[11] ,\u_tester/tx_cnt_reg_n_0_[10] ,\u_tester/tx_cnt_reg_n_0_[9] ,\u_tester/tx_cnt_reg_n_0_[8] }));
  FDRE \u_i2c_master/data_o0_reg[0] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[0] ),
        .Q(i2c_data_o0[0]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o0_reg[1] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[1] ),
        .Q(i2c_data_o0[1]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o0_reg[2] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[2] ),
        .Q(i2c_data_o0[2]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o0_reg[3] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[3] ),
        .Q(i2c_data_o0[3]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o0_reg[4] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[4] ),
        .Q(i2c_data_o0[4]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o0_reg[5] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[5] ),
        .Q(i2c_data_o0[5]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o0_reg[6] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[6] ),
        .Q(i2c_data_o0[6]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o0_reg[7] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[7] ),
        .Q(i2c_data_o0[7]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o1_reg[0] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[0] ),
        .Q(i2c_data_o1[0]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o1_reg[1] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[1] ),
        .Q(i2c_data_o1[1]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o1_reg[2] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[2] ),
        .Q(i2c_data_o1[2]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o1_reg[3] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[3] ),
        .Q(i2c_data_o1[3]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o1_reg[4] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[4] ),
        .Q(i2c_data_o1[4]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o1_reg[5] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[5] ),
        .Q(i2c_data_o1[5]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o1_reg[6] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[6] ),
        .Q(i2c_data_o1[6]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/data_o1_reg[7] 
       (.C(clk),
        .CE(\u_i2c_master/ext_cmd_queued0 ),
        .D(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[7] ),
        .Q(i2c_data_o1[7]),
        .R(\data_o0[7]_i_1__0_n_0 ));
  FDRE \u_i2c_master/error_reg__0 
       (.C(clk),
        .CE(1'b1),
        .D(error__0_i_1_n_0),
        .Q(i2c_error),
        .R(1'b0));
  FDRE \u_i2c_master/ext_cmd_queued_reg 
       (.C(clk),
        .CE(1'b1),
        .D(ext_cmd_queued_i_1_n_0),
        .Q(i2c_busy),
        .R(rst));
  FDRE \u_i2c_master/i2c_addr_s_reg[0] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_addr_s[0]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_addr_s [0]),
        .R(rst));
  FDRE \u_i2c_master/i2c_addr_s_reg[1] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_addr_s[1]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_addr_s [1]),
        .R(rst));
  FDRE \u_i2c_master/i2c_addr_s_reg[2] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_addr_s[2]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_addr_s [2]),
        .R(rst));
  FDRE \u_i2c_master/i2c_addr_s_reg[3] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_addr_s[3]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_addr_s [3]),
        .R(rst));
  FDRE \u_i2c_master/i2c_addr_s_reg[4] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_addr_s[4]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_addr_s [4]),
        .R(rst));
  FDRE \u_i2c_master/i2c_addr_s_reg[5] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_addr_s[5]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_addr_s [5]),
        .R(rst));
  FDRE \u_i2c_master/i2c_addr_s_reg[6] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_addr_s[6]_i_2_n_0 ),
        .Q(\u_i2c_master/i2c_addr_s [6]),
        .R(rst));
  FDRE \u_i2c_master/i2c_cmd_mode_reg[0] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\u_i2c_master/i2c_cmd_mode [0]),
        .Q(\u_i2c_master/i2c_cmd_mode_reg_n_0_[0] ),
        .R(rst));
  FDRE \u_i2c_master/i2c_cmd_mode_reg[1] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\u_i2c_master/i2c_cmd_mode [1]),
        .Q(\u_i2c_master/i2c_cmd_mode_reg_n_0_[1] ),
        .R(rst));
  FDRE \u_i2c_master/i2c_cmd_mode_reg[2] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\u_i2c_master/i2c_cmd_mode [2]),
        .Q(\u_i2c_master/i2c_cmd_mode_reg_n_0_[2] ),
        .R(rst));
  FDRE \u_i2c_master/i2c_cmd_mode_reg[3] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\u_i2c_master/i2c_cmd_mode [3]),
        .Q(\u_i2c_master/i2c_cmd_mode_reg_n_0_[3] ),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i0_reg[0] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i0[0]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i0 [0]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i0_reg[1] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i0[1]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i0 [1]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i0_reg[2] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i0[2]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i0 [2]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i0_reg[3] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i0[3]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i0 [3]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i0_reg[4] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i0[4]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i0 [4]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i0_reg[5] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i0[5]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i0 [5]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i0_reg[6] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i0[6]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i0 [6]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i0_reg[7] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i0[7]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i0 [7]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i1_reg[0] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i1[0]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i1 [0]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i1_reg[1] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i1[1]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i1 [1]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i1_reg[2] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i1[2]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i1 [2]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i1_reg[3] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i1[3]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i1 [3]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i1_reg[4] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i1[4]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i1 [4]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i1_reg[5] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i1[5]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i1 [5]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i1_reg[6] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i1[6]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i1 [6]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i1_reg[7] 
       (.C(clk),
        .CE(\i2c_addr_s[6]_i_1_n_0 ),
        .D(\i2c_data_i1[7]_i_1_n_0 ),
        .Q(\u_i2c_master/i2c_data_i1 [7]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i2_reg[0] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1_n_0 ),
        .D(i2c_data_i2[0]),
        .Q(\u_i2c_master/i2c_data_i2 [0]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i2_reg[1] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1_n_0 ),
        .D(i2c_data_i2[1]),
        .Q(\u_i2c_master/i2c_data_i2 [1]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i2_reg[2] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1_n_0 ),
        .D(i2c_data_i2[2]),
        .Q(\u_i2c_master/i2c_data_i2 [2]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i2_reg[3] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1_n_0 ),
        .D(i2c_data_i2[3]),
        .Q(\u_i2c_master/i2c_data_i2 [3]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i2_reg[4] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1_n_0 ),
        .D(i2c_data_i2[4]),
        .Q(\u_i2c_master/i2c_data_i2 [4]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i2_reg[5] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1_n_0 ),
        .D(i2c_data_i2[5]),
        .Q(\u_i2c_master/i2c_data_i2 [5]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i2_reg[6] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1_n_0 ),
        .D(i2c_data_i2[6]),
        .Q(\u_i2c_master/i2c_data_i2 [6]),
        .R(rst));
  FDRE \u_i2c_master/i2c_data_i2_reg[7] 
       (.C(clk),
        .CE(\i2c_data_i2[7]_i_1_n_0 ),
        .D(i2c_data_i2[7]),
        .Q(\u_i2c_master/i2c_data_i2 [7]),
        .R(rst));
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    \u_i2c_master/last_wr_cache_reg_0_7_0_0 
       (.A0(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .A1(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .A2(\u_i2c_master/st_addr_reg_n_0_[2] ),
        .A3(1'b0),
        .A4(1'b0),
        .D(\u_i2c_master/st_reg_data [0]),
        .O(\u_i2c_master/st_is_dirty0 [0]),
        .WCLK(clk),
        .WE(p_0_in__0));
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    \u_i2c_master/last_wr_cache_reg_0_7_1_1 
       (.A0(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .A1(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .A2(\u_i2c_master/st_addr_reg_n_0_[2] ),
        .A3(1'b0),
        .A4(1'b0),
        .D(\u_i2c_master/st_reg_data [1]),
        .O(\u_i2c_master/st_is_dirty0 [1]),
        .WCLK(clk),
        .WE(p_0_in__0));
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    \u_i2c_master/last_wr_cache_reg_0_7_2_2 
       (.A0(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .A1(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .A2(\u_i2c_master/st_addr_reg_n_0_[2] ),
        .A3(1'b0),
        .A4(1'b0),
        .D(\u_i2c_master/st_reg_data [2]),
        .O(\u_i2c_master/st_is_dirty0 [2]),
        .WCLK(clk),
        .WE(p_0_in__0));
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    \u_i2c_master/last_wr_cache_reg_0_7_3_3 
       (.A0(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .A1(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .A2(\u_i2c_master/st_addr_reg_n_0_[2] ),
        .A3(1'b0),
        .A4(1'b0),
        .D(\u_i2c_master/st_reg_data [3]),
        .O(\u_i2c_master/st_is_dirty0 [3]),
        .WCLK(clk),
        .WE(p_0_in__0));
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    \u_i2c_master/last_wr_cache_reg_0_7_4_4 
       (.A0(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .A1(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .A2(\u_i2c_master/st_addr_reg_n_0_[2] ),
        .A3(1'b0),
        .A4(1'b0),
        .D(\u_i2c_master/st_reg_data [4]),
        .O(\u_i2c_master/st_is_dirty0 [4]),
        .WCLK(clk),
        .WE(p_0_in__0));
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    \u_i2c_master/last_wr_cache_reg_0_7_5_5 
       (.A0(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .A1(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .A2(\u_i2c_master/st_addr_reg_n_0_[2] ),
        .A3(1'b0),
        .A4(1'b0),
        .D(\u_i2c_master/st_reg_data [5]),
        .O(\u_i2c_master/st_is_dirty0 [5]),
        .WCLK(clk),
        .WE(p_0_in__0));
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    \u_i2c_master/last_wr_cache_reg_0_7_6_6 
       (.A0(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .A1(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .A2(\u_i2c_master/st_addr_reg_n_0_[2] ),
        .A3(1'b0),
        .A4(1'b0),
        .D(\u_i2c_master/st_reg_data [6]),
        .O(\u_i2c_master/st_is_dirty0 [6]),
        .WCLK(clk),
        .WE(p_0_in__0));
  (* XILINX_LEGACY_PRIM = "RAM16X1S" *) 
  RAM32X1S #(
    .INIT(32'h00000000)) 
    \u_i2c_master/last_wr_cache_reg_0_7_7_7 
       (.A0(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .A1(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .A2(\u_i2c_master/st_addr_reg_n_0_[2] ),
        .A3(1'b0),
        .A4(1'b0),
        .D(\u_i2c_master/st_reg_data [7]),
        .O(\u_i2c_master/st_is_dirty0 [7]),
        .WCLK(clk),
        .WE(p_0_in__0));
  FDRE \u_i2c_master/prog_cache_entries_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\prog_cache_entries[0]_i_1_n_0 ),
        .Q(\u_i2c_master/prog_cache_entries [0]),
        .R(rst));
  FDRE \u_i2c_master/prog_cache_entries_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\prog_cache_entries[1]_i_1_n_0 ),
        .Q(\u_i2c_master/prog_cache_entries [1]),
        .R(rst));
  FDRE \u_i2c_master/prog_cache_entries_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\prog_cache_entries[2]_i_1_n_0 ),
        .Q(\u_i2c_master/prog_cache_entries [2]),
        .R(rst));
  FDRE \u_i2c_master/prog_cache_entries_reg[3] 
       (.C(clk),
        .CE(1'b1),
        .D(\prog_cache_entries[3]_i_1_n_0 ),
        .Q(\u_i2c_master/prog_cache_entries [3]),
        .R(rst));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "7" *) 
  (* ram_slice_begin = "0" *) 
  (* ram_slice_end = "5" *) 
  RAM32M \u_i2c_master/prog_cache_reg_0_7_0_5 
       (.ADDRA({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRB({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRC({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRD({1'b0,1'b0,i2c_prog_cache_addr}),
        .DIA({prog_cache_reg_0_7_0_5_i_2_n_0,prog_cache_reg_0_7_0_5_i_3_n_0}),
        .DIB({prog_cache_reg_0_7_0_5_i_4_n_0,prog_cache_reg_0_7_0_5_i_2_n_0}),
        .DIC({1'b0,1'b0}),
        .DID({1'b0,1'b0}),
        .DOA({\u_i2c_master/prog_cache_reg_0_7_0_5_n_0 ,\u_i2c_master/prog_cache_reg_0_7_0_5_n_1 }),
        .DOB({\u_i2c_master/prog_cache_reg_0_7_0_5_n_2 ,\u_i2c_master/prog_cache_reg_0_7_0_5_n_3 }),
        .DOC({\u_i2c_master/prog_cache_reg_0_7_0_5_n_4 ,\u_i2c_master/prog_cache_reg_0_7_0_5_n_5 }),
        .DOD(\NLW_u_i2c_master/prog_cache_reg_0_7_0_5_DOD_UNCONNECTED [1:0]),
        .WCLK(clk),
        .WE(prog_cache_reg_0_7_0_5_i_1_n_0));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "7" *) 
  (* ram_slice_begin = "12" *) 
  (* ram_slice_end = "15" *) 
  RAM32M \u_i2c_master/prog_cache_reg_0_7_12_15 
       (.ADDRA({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRB({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRC({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRD({1'b0,1'b0,i2c_prog_cache_addr}),
        .DIA({prog_cache_reg_0_7_12_15_i_1_n_0,1'b0}),
        .DIB({prog_cache_reg_0_7_12_15_i_1_n_0,prog_cache_reg_0_7_12_15_i_1_n_0}),
        .DIC({1'b0,1'b0}),
        .DID({1'b0,1'b0}),
        .DOA({\u_i2c_master/prog_cache_reg_0_7_12_15_n_0 ,\u_i2c_master/prog_cache_reg_0_7_12_15_n_1 }),
        .DOB({\u_i2c_master/st_wr_mode ,\u_i2c_master/prog_cache_reg_0_7_12_15_n_3 }),
        .DOC(\NLW_u_i2c_master/prog_cache_reg_0_7_12_15_DOC_UNCONNECTED [1:0]),
        .DOD(\NLW_u_i2c_master/prog_cache_reg_0_7_12_15_DOD_UNCONNECTED [1:0]),
        .WCLK(clk),
        .WE(prog_cache_reg_0_7_0_5_i_1_n_0));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "7" *) 
  (* ram_slice_begin = "6" *) 
  (* ram_slice_end = "11" *) 
  RAM32M \u_i2c_master/prog_cache_reg_0_7_6_11 
       (.ADDRA({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRB({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRC({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRD({1'b0,1'b0,i2c_prog_cache_addr}),
        .DIA({1'b0,1'b0}),
        .DIB({1'b0,1'b0}),
        .DIC({1'b0,1'b0}),
        .DID({1'b0,1'b0}),
        .DOA({\u_i2c_master/prog_cache_reg_0_7_6_11_n_0 ,\u_i2c_master/prog_cache_reg_0_7_6_11_n_1 }),
        .DOB({\u_i2c_master/prog_cache_reg_0_7_6_11_n_2 ,\u_i2c_master/prog_cache_reg_0_7_6_11_n_3 }),
        .DOC({\u_i2c_master/prog_cache_reg_0_7_6_11_n_4 ,\u_i2c_master/prog_cache_reg_0_7_6_11_n_5 }),
        .DOD(\NLW_u_i2c_master/prog_cache_reg_0_7_6_11_DOD_UNCONNECTED [1:0]),
        .WCLK(clk),
        .WE(prog_cache_reg_0_7_0_5_i_1_n_0));
  FDRE \u_i2c_master/st_addr_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\st_addr[0]_i_1_n_0 ),
        .Q(\u_i2c_master/st_addr_reg_n_0_[0] ),
        .R(rst));
  FDRE \u_i2c_master/st_addr_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\st_addr[1]_i_1_n_0 ),
        .Q(\u_i2c_master/st_addr_reg_n_0_[1] ),
        .R(rst));
  FDRE \u_i2c_master/st_addr_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\st_addr[2]_i_1_n_0 ),
        .Q(\u_i2c_master/st_addr_reg_n_0_[2] ),
        .R(rst));
  FDRE \u_i2c_master/state_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\state[0]_i_1_n_0 ),
        .Q(\u_i2c_master/state_reg_n_0_[0] ),
        .R(rst));
  FDRE \u_i2c_master/state_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\state[1]_i_1_n_0 ),
        .Q(\u_i2c_master/state_reg_n_0_[1] ),
        .R(rst));
  FDRE \u_i2c_master/state_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\state[2]_i_1_n_0 ),
        .Q(\u_i2c_master/state_reg_n_0_[2] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "ST_WRITE_DATA1:0000,ST_WRITE_DATA2:0010,ST_WRITE_DATA0:0011,ST_WAIT_INACTIVE:0101,ST_WR_CMD_WR0:1000,ST_WAIT_CMD:0111,ST_WR_CMD_STOP:0001,ST_WR_CMD_RD1:1001,ST_READ_DATA1:1010,ST_READ_DATA0:1011,iSTATE:0110,ST_WR_CMD_RD0:0100" *) 
  FDSE \u_i2c_master/u_i2c_master/FSM_sequential_state_reg[0] 
       (.C(clk),
        .CE(\FSM_sequential_state[3]_i_1_n_0 ),
        .D(\FSM_sequential_state[0]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/state [0]),
        .S(rst));
  (* FSM_ENCODED_STATES = "ST_WRITE_DATA1:0000,ST_WRITE_DATA2:0010,ST_WRITE_DATA0:0011,ST_WAIT_INACTIVE:0101,ST_WR_CMD_WR0:1000,ST_WAIT_CMD:0111,ST_WR_CMD_STOP:0001,ST_WR_CMD_RD1:1001,ST_READ_DATA1:1010,ST_READ_DATA0:1011,iSTATE:0110,ST_WR_CMD_RD0:0100" *) 
  FDSE \u_i2c_master/u_i2c_master/FSM_sequential_state_reg[1] 
       (.C(clk),
        .CE(\FSM_sequential_state[3]_i_1_n_0 ),
        .D(\FSM_sequential_state[1]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/state [1]),
        .S(rst));
  (* FSM_ENCODED_STATES = "ST_WRITE_DATA1:0000,ST_WRITE_DATA2:0010,ST_WRITE_DATA0:0011,ST_WAIT_INACTIVE:0101,ST_WR_CMD_WR0:1000,ST_WAIT_CMD:0111,ST_WR_CMD_STOP:0001,ST_WR_CMD_RD1:1001,ST_READ_DATA1:1010,ST_READ_DATA0:1011,iSTATE:0110,ST_WR_CMD_RD0:0100" *) 
  FDSE \u_i2c_master/u_i2c_master/FSM_sequential_state_reg[2] 
       (.C(clk),
        .CE(\FSM_sequential_state[3]_i_1_n_0 ),
        .D(\FSM_sequential_state[2]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/state [2]),
        .S(rst));
  (* FSM_ENCODED_STATES = "ST_WRITE_DATA1:0000,ST_WRITE_DATA2:0010,ST_WRITE_DATA0:0011,ST_WAIT_INACTIVE:0101,ST_WR_CMD_WR0:1000,ST_WAIT_CMD:0111,ST_WR_CMD_STOP:0001,ST_WR_CMD_RD1:1001,ST_READ_DATA1:1010,ST_READ_DATA0:1011,iSTATE:0110,ST_WR_CMD_RD0:0100" *) 
  FDRE \u_i2c_master/u_i2c_master/FSM_sequential_state_reg[3] 
       (.C(clk),
        .CE(\FSM_sequential_state[3]_i_1_n_0 ),
        .D(\FSM_sequential_state[3]_i_2_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/state [3]),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/cmd_mode_r_reg[0] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/cmd_mode_r ),
        .D(\u_i2c_master/i2c_cmd_mode_reg_n_0_[0] ),
        .Q(\u_i2c_master/u_i2c_master/cmd_mode_r_reg_n_0_[0] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/cmd_mode_r_reg[1] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/cmd_mode_r ),
        .D(\u_i2c_master/i2c_cmd_mode_reg_n_0_[1] ),
        .Q(\u_i2c_master/u_i2c_master/p_0_in7_in ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/cmd_mode_r_reg[2] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/cmd_mode_r ),
        .D(\u_i2c_master/i2c_cmd_mode_reg_n_0_[2] ),
        .Q(\u_i2c_master/u_i2c_master/p_1_in ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/cmd_mode_r_reg[3] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/cmd_mode_r ),
        .D(\u_i2c_master/i2c_cmd_mode_reg_n_0_[3] ),
        .Q(\u_i2c_master/u_i2c_master/p_4_in ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o0_reg[0] 
       (.C(clk),
        .CE(\data_o0[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[0] ),
        .Q(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[0] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o0_reg[1] 
       (.C(clk),
        .CE(\data_o0[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[1] ),
        .Q(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[1] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o0_reg[2] 
       (.C(clk),
        .CE(\data_o0[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[2] ),
        .Q(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[2] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o0_reg[3] 
       (.C(clk),
        .CE(\data_o0[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[3] ),
        .Q(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[3] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o0_reg[4] 
       (.C(clk),
        .CE(\data_o0[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[4] ),
        .Q(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[4] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o0_reg[5] 
       (.C(clk),
        .CE(\data_o0[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[5] ),
        .Q(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[5] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o0_reg[6] 
       (.C(clk),
        .CE(\data_o0[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[6] ),
        .Q(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[6] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o0_reg[7] 
       (.C(clk),
        .CE(\data_o0[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[7] ),
        .Q(\u_i2c_master/u_i2c_master/data_o0_reg_n_0_[7] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o1_reg[0] 
       (.C(clk),
        .CE(\data_o1[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[0] ),
        .Q(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[0] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o1_reg[1] 
       (.C(clk),
        .CE(\data_o1[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[1] ),
        .Q(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[1] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o1_reg[2] 
       (.C(clk),
        .CE(\data_o1[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[2] ),
        .Q(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[2] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o1_reg[3] 
       (.C(clk),
        .CE(\data_o1[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[3] ),
        .Q(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[3] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o1_reg[4] 
       (.C(clk),
        .CE(\data_o1[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[4] ),
        .Q(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[4] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o1_reg[5] 
       (.C(clk),
        .CE(\data_o1[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[5] ),
        .Q(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[5] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o1_reg[6] 
       (.C(clk),
        .CE(\data_o1[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[6] ),
        .Q(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[6] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/data_o1_reg[7] 
       (.C(clk),
        .CE(\data_o1[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[7] ),
        .Q(\u_i2c_master/u_i2c_master/data_o1_reg_n_0_[7] ),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/error_reg 
       (.C(clk),
        .CE(1'b1),
        .D(error_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/error_reg_n_0 ),
        .R(1'b0));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDSE #(
    .INIT(1'b1)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[0] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[0]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[0] ),
        .S(rst));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[10] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[10]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[10] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[11] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[11]_i_2_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[11] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[1] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[1]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[1] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[2] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[2]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[2] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[3] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[3]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[3] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[4] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[4]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[4] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[5] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[5]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[5] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[6] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[6]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[6] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[7] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[7]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[7] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[8] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[8]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[8] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "STATE_IDLE:000000000001,STATE_WRITE_3:010000000000,STATE_WRITE_1:000100000000,STATE_WRITE_2:001000000000,STATE_ADDRESS_2:000000001000,STATE_ADDRESS_1:000000000100,STATE_START_WAIT:000000000010,STATE_START:000001000000,STATE_ACTIVE_READ:000000100000,STATE_STOP:000010000000,STATE_READ:000000010000,STATE_ACTIVE_WRITE:100000000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg[9] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/p_4_in ),
        .D(\FSM_onehot_state_reg[9]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/FSM_onehot_state_reg_reg_n_0_[9] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "PHY_STATE_STOP_2:1110,PHY_STATE_STOP_1:1101,PHY_STATE_READ_BIT_4:1100,PHY_STATE_REPEATED_START_2:0101,PHY_STATE_REPEATED_START_1:0100,PHY_STATE_ACTIVE:0011,PHY_STATE_READ_BIT_3:1011,PHY_STATE_IDLE:0000,PHY_STATE_READ_BIT_2:1010,PHY_STATE_WRITE_BIT_2:0111,PHY_STATE_WRITE_BIT_1:0110,PHY_STATE_READ_BIT_1:1001,PHY_STATE_WRITE_BIT_3:1000,PHY_STATE_START_2:0010,PHY_STATE_START_1:0001,PHY_STATE_STOP_3:1111" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_sequential_phy_state_reg_reg[0] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_next ),
        .D(\FSM_sequential_phy_state_reg[0]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [0]),
        .R(rst));
  (* FSM_ENCODED_STATES = "PHY_STATE_STOP_2:1110,PHY_STATE_STOP_1:1101,PHY_STATE_READ_BIT_4:1100,PHY_STATE_REPEATED_START_2:0101,PHY_STATE_REPEATED_START_1:0100,PHY_STATE_ACTIVE:0011,PHY_STATE_READ_BIT_3:1011,PHY_STATE_IDLE:0000,PHY_STATE_READ_BIT_2:1010,PHY_STATE_WRITE_BIT_2:0111,PHY_STATE_WRITE_BIT_1:0110,PHY_STATE_READ_BIT_1:1001,PHY_STATE_WRITE_BIT_3:1000,PHY_STATE_START_2:0010,PHY_STATE_START_1:0001,PHY_STATE_STOP_3:1111" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_sequential_phy_state_reg_reg[1] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_next ),
        .D(\FSM_sequential_phy_state_reg[1]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [1]),
        .R(rst));
  (* FSM_ENCODED_STATES = "PHY_STATE_STOP_2:1110,PHY_STATE_STOP_1:1101,PHY_STATE_READ_BIT_4:1100,PHY_STATE_REPEATED_START_2:0101,PHY_STATE_REPEATED_START_1:0100,PHY_STATE_ACTIVE:0011,PHY_STATE_READ_BIT_3:1011,PHY_STATE_IDLE:0000,PHY_STATE_READ_BIT_2:1010,PHY_STATE_WRITE_BIT_2:0111,PHY_STATE_WRITE_BIT_1:0110,PHY_STATE_READ_BIT_1:1001,PHY_STATE_WRITE_BIT_3:1000,PHY_STATE_START_2:0010,PHY_STATE_START_1:0001,PHY_STATE_STOP_3:1111" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_sequential_phy_state_reg_reg[2] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_next ),
        .D(\FSM_sequential_phy_state_reg[2]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [2]),
        .R(rst));
  (* FSM_ENCODED_STATES = "PHY_STATE_STOP_2:1110,PHY_STATE_STOP_1:1101,PHY_STATE_READ_BIT_4:1100,PHY_STATE_REPEATED_START_2:0101,PHY_STATE_REPEATED_START_1:0100,PHY_STATE_ACTIVE:0011,PHY_STATE_READ_BIT_3:1011,PHY_STATE_IDLE:0000,PHY_STATE_READ_BIT_2:1010,PHY_STATE_WRITE_BIT_2:0111,PHY_STATE_WRITE_BIT_1:0110,PHY_STATE_READ_BIT_1:1001,PHY_STATE_WRITE_BIT_3:1000,PHY_STATE_START_2:0010,PHY_STATE_START_1:0001,PHY_STATE_STOP_3:1111" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/FSM_sequential_phy_state_reg_reg[3] 
       (.C(clk),
        .CE(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_next ),
        .D(\FSM_sequential_phy_state_reg[3]_i_2_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/phy_state_reg [3]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/addr_reg_reg[0] 
       (.C(clk),
        .CE(\addr_reg[6]_i_1_n_0 ),
        .D(\u_i2c_master/i2c_addr_s [0]),
        .Q(\u_i2c_master/addr_reg [0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/addr_reg_reg[1] 
       (.C(clk),
        .CE(\addr_reg[6]_i_1_n_0 ),
        .D(\u_i2c_master/i2c_addr_s [1]),
        .Q(\u_i2c_master/addr_reg [1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/addr_reg_reg[2] 
       (.C(clk),
        .CE(\addr_reg[6]_i_1_n_0 ),
        .D(\u_i2c_master/i2c_addr_s [2]),
        .Q(\u_i2c_master/addr_reg [2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/addr_reg_reg[3] 
       (.C(clk),
        .CE(\addr_reg[6]_i_1_n_0 ),
        .D(\u_i2c_master/i2c_addr_s [3]),
        .Q(\u_i2c_master/addr_reg [3]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/addr_reg_reg[4] 
       (.C(clk),
        .CE(\addr_reg[6]_i_1_n_0 ),
        .D(\u_i2c_master/i2c_addr_s [4]),
        .Q(\u_i2c_master/addr_reg [4]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/addr_reg_reg[5] 
       (.C(clk),
        .CE(\addr_reg[6]_i_1_n_0 ),
        .D(\u_i2c_master/i2c_addr_s [5]),
        .Q(\u_i2c_master/addr_reg [5]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/addr_reg_reg[6] 
       (.C(clk),
        .CE(\addr_reg[6]_i_1_n_0 ),
        .D(\u_i2c_master/i2c_addr_s [6]),
        .Q(\u_i2c_master/addr_reg [6]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/bit_count_reg_reg[0] 
       (.C(clk),
        .CE(\bit_count_reg[3]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/bit_count_next [0]),
        .Q(\u_i2c_master/bit_count_reg [0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/bit_count_reg_reg[1] 
       (.C(clk),
        .CE(\bit_count_reg[3]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/bit_count_next [1]),
        .Q(\u_i2c_master/bit_count_reg [1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/bit_count_reg_reg[2] 
       (.C(clk),
        .CE(\bit_count_reg[3]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/bit_count_next [2]),
        .Q(\u_i2c_master/bit_count_reg [2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/bit_count_reg_reg[3] 
       (.C(clk),
        .CE(\bit_count_reg[3]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/bit_count_next [3]),
        .Q(\u_i2c_master/bit_count_reg [3]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/bus_active_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(bus_active_reg_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/bus_active_reg_reg_n_0 ),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/busy_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/busy_reg0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/busy_reg_reg_n_0 ),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/cmd_ready_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(cmd_ready_reg_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/cmd_ready0 ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_in_ready_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(data_in_ready_reg_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/data_in_ready0 ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_out_last_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(data_out_last_reg_i_1_n_0),
        .Q(\u_i2c_master/data_out_last_reg ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_out_reg_reg[0] 
       (.C(clk),
        .CE(\data_out_reg[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [0]),
        .Q(\u_i2c_master/data_out_reg [0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_out_reg_reg[1] 
       (.C(clk),
        .CE(\data_out_reg[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [1]),
        .Q(\u_i2c_master/data_out_reg [1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_out_reg_reg[2] 
       (.C(clk),
        .CE(\data_out_reg[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [2]),
        .Q(\u_i2c_master/data_out_reg [2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_out_reg_reg[3] 
       (.C(clk),
        .CE(\data_out_reg[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [3]),
        .Q(\u_i2c_master/data_out_reg [3]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_out_reg_reg[4] 
       (.C(clk),
        .CE(\data_out_reg[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [4]),
        .Q(\u_i2c_master/data_out_reg [4]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_out_reg_reg[5] 
       (.C(clk),
        .CE(\data_out_reg[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [5]),
        .Q(\u_i2c_master/data_out_reg [5]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_out_reg_reg[6] 
       (.C(clk),
        .CE(\data_out_reg[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [6]),
        .Q(\u_i2c_master/data_out_reg [6]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_out_reg_reg[7] 
       (.C(clk),
        .CE(\data_out_reg[7]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [7]),
        .Q(\u_i2c_master/data_out_reg [7]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_out_valid_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(data_out_valid_reg_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/data_out_valid0 ),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_reg_reg[0] 
       (.C(clk),
        .CE(\data_reg[7]_i_1_n_0 ),
        .D(\data_reg[0]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_reg_reg[1] 
       (.C(clk),
        .CE(\data_reg[7]_i_1_n_0 ),
        .D(\data_reg[1]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [2]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_reg_reg[2] 
       (.C(clk),
        .CE(\data_reg[7]_i_1_n_0 ),
        .D(\data_reg[2]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [3]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_reg_reg[3] 
       (.C(clk),
        .CE(\data_reg[7]_i_1_n_0 ),
        .D(\data_reg[3]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [4]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_reg_reg[4] 
       (.C(clk),
        .CE(\data_reg[7]_i_1_n_0 ),
        .D(\data_reg[4]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [5]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_reg_reg[5] 
       (.C(clk),
        .CE(\data_reg[7]_i_1_n_0 ),
        .D(\data_reg[5]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [6]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_reg_reg[6] 
       (.C(clk),
        .CE(\data_reg[7]_i_1_n_0 ),
        .D(\data_reg[6]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [7]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/data_reg_reg[7] 
       (.C(clk),
        .CE(\data_reg[7]_i_1_n_0 ),
        .D(\data_reg[7]_i_2_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/data_reg_reg_n_0_[7] ),
        .R(1'b0));
  CARRY4 \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry 
       (.CI(1'b0),
        .CO({\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry_n_0 ,\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry_n_1 ,\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry_n_2 ,\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry_n_3 }),
        .CYINIT(\u_i2c_master/delay_reg [0]),
        .DI(\u_i2c_master/delay_reg [4:1]),
        .O(\u_i2c_master/delay_next0 [4:1]),
        .S({delay_next0_carry_i_1_n_0,delay_next0_carry_i_2_n_0,delay_next0_carry_i_3_n_0,delay_next0_carry_i_4_n_0}));
  CARRY4 \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__0 
       (.CI(\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry_n_0 ),
        .CO({\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__0_n_0 ,\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__0_n_1 ,\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__0_n_2 ,\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__0_n_3 }),
        .CYINIT(1'b0),
        .DI(\u_i2c_master/delay_reg [8:5]),
        .O(\u_i2c_master/delay_next0 [8:5]),
        .S({delay_next0_carry__0_i_1_n_0,delay_next0_carry__0_i_2_n_0,delay_next0_carry__0_i_3_n_0,delay_next0_carry__0_i_4_n_0}));
  CARRY4 \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__1 
       (.CI(\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__0_n_0 ),
        .CO({\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__1_n_0 ,\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__1_n_1 ,\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__1_n_2 ,\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__1_n_3 }),
        .CYINIT(1'b0),
        .DI(\u_i2c_master/delay_reg [12:9]),
        .O(\u_i2c_master/delay_next0 [12:9]),
        .S({delay_next0_carry__1_i_1_n_0,delay_next0_carry__1_i_2_n_0,delay_next0_carry__1_i_3_n_0,delay_next0_carry__1_i_4_n_0}));
  CARRY4 \u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__2 
       (.CI(\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__1_n_0 ),
        .CO({\NLW_u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__2_CO_UNCONNECTED [3],\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__2_n_1 ,\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__2_n_2 ,\u_i2c_master/u_i2c_master/u_i2c_master/delay_next0_carry__2_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,\u_i2c_master/delay_reg [15:13]}),
        .O(\u_i2c_master/delay_next0 [16:13]),
        .S({delay_next0_carry__2_i_1_n_0,delay_next0_carry__2_i_2_n_0,delay_next0_carry__2_i_3_n_0,delay_next0_carry__2_i_4_n_0}));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[0] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[0]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [0]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[10] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[10]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [10]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[11] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[11]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [11]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[12] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[12]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [12]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[13] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[13]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [13]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[14] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[14]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [14]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[15] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[15]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [15]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[16] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[16]_i_2_n_0 ),
        .Q(\u_i2c_master/delay_reg [16]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[1] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[1]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [1]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[2] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[2]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [2]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[3] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[3]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [3]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[4] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[4]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [4]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[5] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[5]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [5]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[6] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[6]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [6]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[7] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[7]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [7]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[8] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[8]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [8]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_reg_reg[9] 
       (.C(clk),
        .CE(\delay_reg[16]_i_1_n_0 ),
        .D(\delay_reg[9]_i_1_n_0 ),
        .Q(\u_i2c_master/delay_reg [9]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/delay_scl_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(delay_scl_reg_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/delay_scl_reg ),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/last_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(last_reg_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/last_reg ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b1)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/last_sda_i_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/u_i2c_master/u_i2c_master/sda_i_reg ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/last_sda_i_reg ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/missed_ack_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(missed_ack_reg_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/missed_ack_reg_reg_n_0 ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/mode_ping_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(mode_ping_reg_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/mode_ping_reg ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/mode_read_reg_reg 
       (.C(clk),
        .CE(\addr_reg[6]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/cmd_read0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/mode_read_reg ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg_reg 
       (.C(clk),
        .CE(\addr_reg[6]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/cmd_stop0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/mode_stop_reg ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/mode_write_multiple_reg_reg 
       (.C(clk),
        .CE(\addr_reg[6]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/cmd_write_multiple0 ),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/mode_write_multiple_reg ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/phy_rx_data_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(phy_rx_data_reg_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/p_0_in [0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b1)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/scl_i_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(rfmod_in[11]),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/scl_i_reg ),
        .R(1'b0));
  FDSE #(
    .INIT(1'b1)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/scl_o_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(scl_o_reg_i_1_n_0),
        .Q(scl_t),
        .S(rst));
  FDRE #(
    .INIT(1'b1)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/sda_i_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(rfmod_in[10]),
        .Q(\u_i2c_master/u_i2c_master/u_i2c_master/sda_i_reg ),
        .R(1'b0));
  FDSE #(
    .INIT(1'b1)) 
    \u_i2c_master/u_i2c_master/u_i2c_master/sda_o_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(sda_o_reg_i_1_n_0),
        .Q(sda_t),
        .S(rst));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg[0] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1_n_0 ),
        .D(mem_read_data_reg0__0[0]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[0] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg[1] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1_n_0 ),
        .D(mem_read_data_reg0__0[1]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[1] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg[2] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1_n_0 ),
        .D(mem_read_data_reg0__0[2]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[2] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg[3] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1_n_0 ),
        .D(mem_read_data_reg0__0[3]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[3] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg[4] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1_n_0 ),
        .D(mem_read_data_reg0__0[4]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[4] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg[5] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1_n_0 ),
        .D(mem_read_data_reg0__0[5]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[5] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg[6] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1_n_0 ),
        .D(mem_read_data_reg0__0[6]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[6] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg[7] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1_n_0 ),
        .D(mem_read_data_reg0__0[7]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[7] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg[8] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1_n_0 ),
        .D(mem_read_data_reg0__0[8]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[8] ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_valid_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(mem_read_data_valid_reg_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_valid_reg ),
        .R(rst));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "3" *) 
  (* ram_slice_begin = "0" *) 
  (* ram_slice_end = "5" *) 
  RAM32M \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_reg_0_3_0_5 
       (.ADDRA({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_addr_reg }),
        .ADDRB({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_addr_reg }),
        .ADDRC({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_addr_reg }),
        .ADDRD({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_addr_reg }),
        .DIA(\u_i2c_master/data_out_reg [1:0]),
        .DIB(\u_i2c_master/data_out_reg [3:2]),
        .DIC(\u_i2c_master/data_out_reg [5:4]),
        .DID({1'b0,1'b0}),
        .DOA(mem_read_data_reg0__0[1:0]),
        .DOB(mem_read_data_reg0__0[3:2]),
        .DOC(mem_read_data_reg0__0[5:4]),
        .DOD(\NLW_u_i2c_master/u_i2c_master/u_rdata_fifo/mem_reg_0_3_0_5_DOD_UNCONNECTED [1:0]),
        .WCLK(clk),
        .WE(write1_out__0));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "3" *) 
  (* ram_slice_begin = "6" *) 
  (* ram_slice_end = "8" *) 
  RAM32M \u_i2c_master/u_i2c_master/u_rdata_fifo/mem_reg_0_3_6_8 
       (.ADDRA({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_addr_reg }),
        .ADDRB({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_addr_reg }),
        .ADDRC({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_addr_reg }),
        .ADDRD({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_addr_reg }),
        .DIA(\u_i2c_master/data_out_reg [7:6]),
        .DIB({1'b0,\u_i2c_master/data_out_last_reg }),
        .DIC({1'b0,1'b0}),
        .DID({1'b0,1'b0}),
        .DOA(mem_read_data_reg0__0[7:6]),
        .DOB({\NLW_u_i2c_master/u_i2c_master/u_rdata_fifo/mem_reg_0_3_6_8_DOB_UNCONNECTED [1],mem_read_data_reg0__0[8]}),
        .DOC(\NLW_u_i2c_master/u_i2c_master/u_rdata_fifo/mem_reg_0_3_6_8_DOC_UNCONNECTED [1:0]),
        .DOD(\NLW_u_i2c_master/u_i2c_master/u_rdata_fifo/mem_reg_0_3_6_8_DOD_UNCONNECTED [1:0]),
        .WCLK(clk),
        .WE(write1_out__0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg[0] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[0] ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[0] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg[1] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[1] ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[1] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg[2] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[2] ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[2] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg[3] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[3] ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[3] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg[4] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[4] ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[4] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg[5] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[5] ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[5] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg[6] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[6] ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[6] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg[7] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[7] ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg_n_0_[7] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_reg_reg[8] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1_n_0 ),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/mem_read_data_reg_reg_n_0_[8] ),
        .Q(\u_i2c_master/u_i2c_master/data_out_last ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/output_axis_tvalid_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(output_axis_tvalid_reg_i_1_n_0),
        .Q(\u_i2c_master/u_i2c_master/data_out_valid ),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/rd_addr_reg_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\rd_addr_reg[0]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_addr_reg [0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/rd_addr_reg_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\rd_addr_reg[1]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_addr_reg [1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\rd_addr_reg[0]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[0] ),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\rd_addr_reg[1]_i_1_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[1] ),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/rd_ptr_next__0 ),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[2] ),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/wr_addr_reg_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_next [0]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_addr_reg [0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/wr_addr_reg_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_next [1]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_addr_reg [1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_next [0]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [0]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_next [1]),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [1]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(wr_ptr_next__0),
        .Q(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [2]),
        .R(rst));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_reg_reg[0] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .D(mem_read_data_reg0[0]),
        .Q(\u_i2c_master/mem_read_data_reg [0]),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_reg_reg[1] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .D(mem_read_data_reg0[1]),
        .Q(\u_i2c_master/mem_read_data_reg [1]),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_reg_reg[2] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .D(mem_read_data_reg0[2]),
        .Q(\u_i2c_master/mem_read_data_reg [2]),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_reg_reg[3] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .D(mem_read_data_reg0[3]),
        .Q(\u_i2c_master/mem_read_data_reg [3]),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_reg_reg[4] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .D(mem_read_data_reg0[4]),
        .Q(\u_i2c_master/mem_read_data_reg [4]),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_reg_reg[5] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .D(mem_read_data_reg0[5]),
        .Q(\u_i2c_master/mem_read_data_reg [5]),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_reg_reg[6] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .D(mem_read_data_reg0[6]),
        .Q(\u_i2c_master/mem_read_data_reg [6]),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_reg_reg[7] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .D(mem_read_data_reg0[7]),
        .Q(\u_i2c_master/mem_read_data_reg [7]),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_reg_reg[8] 
       (.C(clk),
        .CE(\mem_read_data_reg[8]_i_1__0_n_0 ),
        .D(mem_read_data_reg0[8]),
        .Q(\u_i2c_master/mem_read_data_reg [8]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_valid_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(mem_read_data_valid_reg_i_1__0_n_0),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/mem_read_data_valid_reg ),
        .R(rst));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "3" *) 
  (* ram_slice_begin = "0" *) 
  (* ram_slice_end = "5" *) 
  RAM32M \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_reg_0_3_0_5 
       (.ADDRA({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_wdata_fifo/rd_addr_reg }),
        .ADDRB({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_wdata_fifo/rd_addr_reg }),
        .ADDRC({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_wdata_fifo/rd_addr_reg }),
        .ADDRD({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_addr_reg }),
        .DIA({mem_reg_0_3_0_5_i_2_n_0,mem_reg_0_3_0_5_i_3_n_0}),
        .DIB({mem_reg_0_3_0_5_i_4_n_0,mem_reg_0_3_0_5_i_5_n_0}),
        .DIC({mem_reg_0_3_0_5_i_6_n_0,mem_reg_0_3_0_5_i_7_n_0}),
        .DID({1'b0,1'b0}),
        .DOA(mem_read_data_reg0[1:0]),
        .DOB(mem_read_data_reg0[3:2]),
        .DOC(mem_read_data_reg0[5:4]),
        .DOD(\NLW_u_i2c_master/u_i2c_master/u_wdata_fifo/mem_reg_0_3_0_5_DOD_UNCONNECTED [1:0]),
        .WCLK(clk),
        .WE(mem_reg_0_3_0_5_i_1__0_n_0));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "3" *) 
  (* ram_slice_begin = "6" *) 
  (* ram_slice_end = "8" *) 
  RAM32M \u_i2c_master/u_i2c_master/u_wdata_fifo/mem_reg_0_3_6_8 
       (.ADDRA({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_wdata_fifo/rd_addr_reg }),
        .ADDRB({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_wdata_fifo/rd_addr_reg }),
        .ADDRC({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_wdata_fifo/rd_addr_reg }),
        .ADDRD({1'b0,1'b0,1'b0,\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_addr_reg }),
        .DIA({mem_reg_0_3_6_8_i_1_n_0,mem_reg_0_3_6_8_i_2_n_0}),
        .DIB({1'b0,\u_i2c_master/u_i2c_master/data_in_last }),
        .DIC({1'b0,1'b0}),
        .DID({1'b0,1'b0}),
        .DOA(mem_read_data_reg0[7:6]),
        .DOB({\NLW_u_i2c_master/u_i2c_master/u_wdata_fifo/mem_reg_0_3_6_8_DOB_UNCONNECTED [1],mem_read_data_reg0[8]}),
        .DOC(\NLW_u_i2c_master/u_i2c_master/u_wdata_fifo/mem_reg_0_3_6_8_DOC_UNCONNECTED [1:0]),
        .DOD(\NLW_u_i2c_master/u_i2c_master/u_wdata_fifo/mem_reg_0_3_6_8_DOD_UNCONNECTED [1:0]),
        .WCLK(clk),
        .WE(mem_reg_0_3_0_5_i_1__0_n_0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg[0] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1__0_n_0 ),
        .D(\u_i2c_master/mem_read_data_reg [0]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[0] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg[1] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1__0_n_0 ),
        .D(\u_i2c_master/mem_read_data_reg [1]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[1] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg[2] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1__0_n_0 ),
        .D(\u_i2c_master/mem_read_data_reg [2]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[2] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg[3] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1__0_n_0 ),
        .D(\u_i2c_master/mem_read_data_reg [3]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[3] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg[4] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1__0_n_0 ),
        .D(\u_i2c_master/mem_read_data_reg [4]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[4] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg[5] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1__0_n_0 ),
        .D(\u_i2c_master/mem_read_data_reg [5]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[5] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg[6] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1__0_n_0 ),
        .D(\u_i2c_master/mem_read_data_reg [6]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[6] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg[7] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1__0_n_0 ),
        .D(\u_i2c_master/mem_read_data_reg [7]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg_n_0_[7] ),
        .R(1'b0));
  FDRE \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_reg_reg[8] 
       (.C(clk),
        .CE(\output_axis_reg[8]_i_1__0_n_0 ),
        .D(\u_i2c_master/mem_read_data_reg [8]),
        .Q(\u_i2c_master/data_in_last ),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/output_axis_tvalid_reg_reg 
       (.C(clk),
        .CE(1'b1),
        .D(output_axis_tvalid_reg_i_1__0_n_0),
        .Q(\u_i2c_master/u_i2c_master/data_in_valid0 ),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/rd_addr_reg_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\rd_addr_reg[0]_i_1__0_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/rd_addr_reg [0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/rd_addr_reg_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\rd_addr_reg[1]_i_1__0_n_0 ),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/rd_addr_reg [1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/rd_ptr_reg_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\rd_addr_reg[0]_i_1__0_n_0 ),
        .Q(\u_i2c_master/rd_ptr_reg [0]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/rd_ptr_reg_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\rd_addr_reg[1]_i_1__0_n_0 ),
        .Q(\u_i2c_master/rd_ptr_reg [1]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/rd_ptr_reg_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/rd_ptr_next ),
        .Q(\u_i2c_master/rd_ptr_reg [2]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/wr_addr_reg_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_next [0]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_addr_reg [0]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/wr_addr_reg_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_next [1]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_addr_reg [1]),
        .R(1'b0));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_next [0]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [0]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_next [1]),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [1]),
        .R(rst));
  FDRE #(
    .INIT(1'b0)) 
    \u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg[2] 
       (.C(clk),
        .CE(1'b1),
        .D(wr_ptr_next),
        .Q(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [2]),
        .R(rst));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "7" *) 
  (* ram_slice_begin = "0" *) 
  (* ram_slice_end = "5" *) 
  RAM32M \u_i2c_master/wr_cache_reg_0_7_0_5 
       (.ADDRA({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRB({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRC({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRD({1'b0,1'b0,i2c_cache_addr}),
        .DIA({1'b0,wr_cache_reg_0_7_0_5_i_2_n_0}),
        .DIB({1'b0,wr_cache_reg_0_7_0_5_i_3_n_0}),
        .DIC({1'b0,wr_cache_reg_0_7_0_5_i_4_n_0}),
        .DID({1'b0,1'b0}),
        .DOA(\u_i2c_master/st_reg_data [1:0]),
        .DOB(\u_i2c_master/st_reg_data [3:2]),
        .DOC(\u_i2c_master/st_reg_data [5:4]),
        .DOD(\NLW_u_i2c_master/wr_cache_reg_0_7_0_5_DOD_UNCONNECTED [1:0]),
        .WCLK(clk),
        .WE(wr_cache_reg_0_7_0_5_i_1_n_0));
  (* METHODOLOGY_DRC_VIOS = "" *) 
  (* ram_addr_begin = "0" *) 
  (* ram_addr_end = "7" *) 
  (* ram_slice_begin = "6" *) 
  (* ram_slice_end = "7" *) 
  RAM32M \u_i2c_master/wr_cache_reg_0_7_6_7 
       (.ADDRA({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRB({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRC({1'b0,1'b0,\u_i2c_master/st_addr_reg_n_0_[2] ,\u_i2c_master/st_addr_reg_n_0_[1] ,\u_i2c_master/st_addr_reg_n_0_[0] }),
        .ADDRD({1'b0,1'b0,i2c_cache_addr}),
        .DIA({1'b0,wr_cache_reg_0_7_6_7_i_1_n_0}),
        .DIB({1'b0,1'b0}),
        .DIC({1'b0,1'b0}),
        .DID({1'b0,1'b0}),
        .DOA(\u_i2c_master/st_reg_data [7:6]),
        .DOB(\NLW_u_i2c_master/wr_cache_reg_0_7_6_7_DOB_UNCONNECTED [1:0]),
        .DOC(\NLW_u_i2c_master/wr_cache_reg_0_7_6_7_DOC_UNCONNECTED [1:0]),
        .DOD(\NLW_u_i2c_master/wr_cache_reg_0_7_6_7_DOD_UNCONNECTED [1:0]),
        .WCLK(clk),
        .WE(wr_cache_reg_0_7_0_5_i_1_n_0));
  (* FSM_ENCODED_STATES = "ST_SPI_START:0000010,ST_SPI_CYCLES:0000100,ST_SPI_END:0001000,ST_SPI_IDLE:0010000,ST_READY_DONE:0100000,ST_WAIT_ENABLE:0000001,iSTATE:1000000" *) 
  FDSE #(
    .INIT(1'b1)) 
    \u_spi_gpio_exp_master/FSM_onehot_state_reg[0] 
       (.C(clk),
        .CE(\FSM_onehot_state[5]_i_1_n_0 ),
        .D(\FSM_onehot_state[0]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .S(rst));
  (* FSM_ENCODED_STATES = "ST_SPI_START:0000010,ST_SPI_CYCLES:0000100,ST_SPI_END:0001000,ST_SPI_IDLE:0010000,ST_READY_DONE:0100000,ST_WAIT_ENABLE:0000001,iSTATE:1000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_spi_gpio_exp_master/FSM_onehot_state_reg[1] 
       (.C(clk),
        .CE(\FSM_onehot_state[5]_i_1_n_0 ),
        .D(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .Q(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[1] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "ST_SPI_START:0000010,ST_SPI_CYCLES:0000100,ST_SPI_END:0001000,ST_SPI_IDLE:0010000,ST_READY_DONE:0100000,ST_WAIT_ENABLE:0000001,iSTATE:1000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_spi_gpio_exp_master/FSM_onehot_state_reg[2] 
       (.C(clk),
        .CE(\FSM_onehot_state[5]_i_1_n_0 ),
        .D(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[1] ),
        .Q(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "ST_SPI_START:0000010,ST_SPI_CYCLES:0000100,ST_SPI_END:0001000,ST_SPI_IDLE:0010000,ST_READY_DONE:0100000,ST_WAIT_ENABLE:0000001,iSTATE:1000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_spi_gpio_exp_master/FSM_onehot_state_reg[3] 
       (.C(clk),
        .CE(\FSM_onehot_state[5]_i_1_n_0 ),
        .D(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .Q(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[3] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "ST_SPI_START:0000010,ST_SPI_CYCLES:0000100,ST_SPI_END:0001000,ST_SPI_IDLE:0010000,ST_READY_DONE:0100000,ST_WAIT_ENABLE:0000001,iSTATE:1000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_spi_gpio_exp_master/FSM_onehot_state_reg[4] 
       (.C(clk),
        .CE(\FSM_onehot_state[5]_i_1_n_0 ),
        .D(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[3] ),
        .Q(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .R(rst));
  (* FSM_ENCODED_STATES = "ST_SPI_START:0000010,ST_SPI_CYCLES:0000100,ST_SPI_END:0001000,ST_SPI_IDLE:0010000,ST_READY_DONE:0100000,ST_WAIT_ENABLE:0000001,iSTATE:1000000" *) 
  FDRE #(
    .INIT(1'b0)) 
    \u_spi_gpio_exp_master/FSM_onehot_state_reg[5] 
       (.C(clk),
        .CE(\FSM_onehot_state[5]_i_1_n_0 ),
        .D(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .Q(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[5] ),
        .R(rst));
  CARRY4 \u_spi_gpio_exp_master/dat_cache1_carry 
       (.CI(1'b0),
        .CO({\u_spi_gpio_exp_master/dat_cache1_carry_n_0 ,\u_spi_gpio_exp_master/dat_cache1_carry_n_1 ,\u_spi_gpio_exp_master/dat_cache1_carry_n_2 ,\u_spi_gpio_exp_master/dat_cache1_carry_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b1,1'b1,1'b1,1'b1}),
        .O(\NLW_u_spi_gpio_exp_master/dat_cache1_carry_O_UNCONNECTED [3:0]),
        .S({dat_cache1_carry_i_1_n_0,dat_cache1_carry_i_2_n_0,dat_cache1_carry_i_3_n_0,dat_cache1_carry_i_4_n_0}));
  CARRY4 \u_spi_gpio_exp_master/dat_cache1_carry__0 
       (.CI(\u_spi_gpio_exp_master/dat_cache1_carry_n_0 ),
        .CO({\u_spi_gpio_exp_master/dat_cache1_carry__0_n_0 ,\u_spi_gpio_exp_master/dat_cache1_carry__0_n_1 ,\u_spi_gpio_exp_master/dat_cache1_carry__0_n_2 ,\u_spi_gpio_exp_master/dat_cache1_carry__0_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b1,1'b1,1'b1,1'b1}),
        .O(\NLW_u_spi_gpio_exp_master/dat_cache1_carry__0_O_UNCONNECTED [3:0]),
        .S({dat_cache1_carry__0_i_1_n_0,dat_cache1_carry__0_i_2_n_0,dat_cache1_carry__0_i_3_n_0,dat_cache1_carry__0_i_4_n_0}));
  CARRY4 \u_spi_gpio_exp_master/dat_cache1_carry__1 
       (.CI(\u_spi_gpio_exp_master/dat_cache1_carry__0_n_0 ),
        .CO({\NLW_u_spi_gpio_exp_master/dat_cache1_carry__1_CO_UNCONNECTED [3],\u_spi_gpio_exp_master/dat_cache1_carry__1_n_1 ,\u_spi_gpio_exp_master/dat_cache1_carry__1_n_2 ,\u_spi_gpio_exp_master/dat_cache1_carry__1_n_3 }),
        .CYINIT(1'b0),
        .DI({1'b0,1'b1,1'b1,1'b1}),
        .O(\NLW_u_spi_gpio_exp_master/dat_cache1_carry__1_O_UNCONNECTED [3:0]),
        .S({1'b0,dat_cache1_carry__1_i_1_n_0,dat_cache1_carry__1_i_2_n_0,dat_cache1_carry__1_i_3_n_0}));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[0] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[0]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[0] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[10] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[10]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[10] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[11] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[11]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[11] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[12] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[12]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[12] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[13] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[13]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[13] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[14] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[14]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[14] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[15] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[15]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[15] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[16] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[16]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[16] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[17] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[17]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[17] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[18] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[18]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[18] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[19] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[19]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[19] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[1] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[1]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[1] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[20] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[20]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[20] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[21] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[21]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[21] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[22] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[22]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[22] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[23] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[23]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[23] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[24] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[24]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[24] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[25] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[25]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[25] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[26] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[26]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[26] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[27] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[27]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[27] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[28] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[28]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[28] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[29] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[29]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[29] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[2] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[2]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[2] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[30] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[30]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[30] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[31] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[31]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[31] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[3] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[3]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[3] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[4] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[4]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[4] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[5] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[5]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[5] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[6] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[6]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[6] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[7] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[7]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[7] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[8] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[8]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[8] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dat_cache_reg[9] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/dat_cache ),
        .D(gpio_exp_out[9]),
        .Q(\u_spi_gpio_exp_master/dat_cache_reg_n_0_[9] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[0] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [1]),
        .Q(\^cbrs_rev ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[10] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [11]),
        .Q(gpio_exp_rb[10]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[11] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [12]),
        .Q(gpio_exp_rb[11]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[12] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [13]),
        .Q(gpio_exp_rb[12]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[13] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [14]),
        .Q(gpio_exp_rb[13]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[14] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [15]),
        .Q(gpio_exp_rb[14]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[15] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [16]),
        .Q(gpio_exp_rb[15]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[16] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [17]),
        .Q(gpio_exp_rb[16]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[17] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [18]),
        .Q(gpio_exp_rb[17]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[18] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [19]),
        .Q(gpio_exp_rb[18]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[19] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [20]),
        .Q(gpio_exp_rb[19]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[1] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [2]),
        .Q(gpio_exp_rb[1]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[20] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [21]),
        .Q(gpio_exp_rb[20]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[21] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [22]),
        .Q(gpio_exp_rb[21]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[22] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [23]),
        .Q(gpio_exp_rb[22]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[23] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [24]),
        .Q(gpio_exp_rb[23]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[24] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [25]),
        .Q(gpio_exp_rb[24]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[25] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [26]),
        .Q(gpio_exp_rb[25]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[26] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [27]),
        .Q(gpio_exp_rb[26]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[27] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [28]),
        .Q(gpio_exp_rb[27]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[28] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [29]),
        .Q(gpio_exp_rb[28]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[29] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [30]),
        .Q(gpio_exp_rb[29]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[2] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [3]),
        .Q(gpio_exp_rb[2]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[30] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [31]),
        .Q(gpio_exp_rb[30]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[31] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/r_inbus_reg_n_0_[31] ),
        .Q(gpio_exp_rb[31]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[3] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [4]),
        .Q(gpio_exp_rb[3]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[4] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [5]),
        .Q(gpio_exp_rb[4]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[5] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [6]),
        .Q(gpio_exp_rb[5]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[6] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [7]),
        .Q(gpio_exp_rb[6]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[7] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [8]),
        .Q(gpio_exp_rb[7]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[8] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [9]),
        .Q(gpio_exp_rb[8]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/dato_reg[9] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[4] ),
        .D(\u_spi_gpio_exp_master/in15 [10]),
        .Q(gpio_exp_rb[9]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/en_reg 
       (.C(clk),
        .CE(1'b1),
        .D(en_i_1_n_0),
        .Q(gpio_exp_busy),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[0] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[0]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [1]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[10] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[10]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [11]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[11] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[11]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [12]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[12] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[12]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [13]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[13] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[13]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [14]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[14] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[14]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [15]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[15] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[15]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [16]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[16] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[16]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [17]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[17] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[17]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [18]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[18] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[18]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [19]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[19] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[19]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [20]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[1] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[1]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [2]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[20] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[20]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [21]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[21] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[21]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [22]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[22] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[22]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [23]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[23] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[23]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [24]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[24] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[24]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [25]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[25] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[25]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [26]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[26] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[26]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [27]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[27] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[27]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [28]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[28] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[28]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [29]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[29] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[29]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [30]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[2] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[2]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [3]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[30] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[30]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [31]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[31] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[31]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/r_inbus_reg_n_0_[31] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[3] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[3]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [4]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[4] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[4]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [5]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[5] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[5]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [6]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[6] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[6]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [7]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[7] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[7]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [8]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[8] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[8]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [9]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/r_inbus_reg[9] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\r_inbus[9]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in15 [10]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_cnt_reg[0] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\spi_bit_cnt[0]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[0] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_cnt_reg[1] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\spi_bit_cnt[1]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[1] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_cnt_reg[2] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\spi_bit_cnt[2]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[2] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_cnt_reg[3] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\spi_bit_cnt[3]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[3] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_cnt_reg[4] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/r_inbus ),
        .D(\spi_bit_cnt[4]_i_2_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_cnt_reg_n_0_[4] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_div_reg[0] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/spi_bit_div ),
        .D(\spi_bit_div[0]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[0] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_div_reg[1] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/spi_bit_div ),
        .D(\spi_bit_div[1]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[1] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_div_reg[2] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/spi_bit_div ),
        .D(\spi_bit_div[2]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[2] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_div_reg[3] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/spi_bit_div ),
        .D(\spi_bit_div[3]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[3] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_div_reg[4] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/spi_bit_div ),
        .D(\spi_bit_div[4]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[4] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_div_reg[5] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/spi_bit_div ),
        .D(\spi_bit_div[5]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[5] ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_bit_div_reg[6] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/spi_bit_div ),
        .D(\spi_bit_div[6]_i_2_n_0 ),
        .Q(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[6] ),
        .R(rst));
  FDSE \u_spi_gpio_exp_master/spi_cs_n_reg 
       (.C(clk),
        .CE(1'b1),
        .D(spi_cs_n_i_1_n_0),
        .Q(spi_cs_n),
        .S(rst));
  FDRE \u_spi_gpio_exp_master/spi_mosi_reg 
       (.C(clk),
        .CE(1'b1),
        .D(spi_mosi_i_1_n_0),
        .Q(spi_mosi),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/spi_sclk_reg 
       (.C(clk),
        .CE(1'b1),
        .D(spi_sclk_i_1_n_0),
        .Q(spi_sclk),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[0] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[0]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [1]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[10] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[10]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [11]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[11] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[11]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [12]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[12] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[12]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [13]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[13] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[13]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [14]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[14] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[14]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [15]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[15] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[15]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [16]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[16] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[16]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [17]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[17] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[17]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [18]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[18] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[18]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [19]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[19] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[19]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [20]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[1] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[1]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [2]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[20] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[20]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [21]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[21] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[21]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [22]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[22] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[22]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [23]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[23] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[23]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [24]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[24] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[24]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [25]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[25] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[25]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [26]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[26] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[26]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [27]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[27] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[27]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [28]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[28] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[28]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [29]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[29] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[29]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [30]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[2] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[2]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [3]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[30] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[30]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [31]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[31] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[31]_i_2_n_0 ),
        .Q(\u_spi_gpio_exp_master/p_0_in ),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[3] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[3]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [4]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[4] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[4]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [5]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[5] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[5]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [6]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[6] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[6]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [7]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[7] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[7]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [8]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[8] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[8]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [9]),
        .R(rst));
  FDRE \u_spi_gpio_exp_master/w_outbus_reg[9] 
       (.C(clk),
        .CE(\u_spi_gpio_exp_master/w_outbus ),
        .D(\w_outbus[9]_i_1_n_0 ),
        .Q(\u_spi_gpio_exp_master/in17 [10]),
        .R(rst));
  FDRE \u_tester/count_reg[0] 
       (.C(clk),
        .CE(test_en),
        .D(p_0_in__1[0]),
        .Q(\u_tester/count_reg [0]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[1] 
       (.C(clk),
        .CE(test_en),
        .D(p_0_in__1[1]),
        .Q(\u_tester/count_reg [1]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[2] 
       (.C(clk),
        .CE(test_en),
        .D(p_0_in__1[2]),
        .Q(\u_tester/count_reg [2]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[3] 
       (.C(clk),
        .CE(test_en),
        .D(p_0_in__1[3]),
        .Q(\u_tester/count_reg [3]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[4] 
       (.C(clk),
        .CE(test_en),
        .D(p_0_in__1[4]),
        .Q(\u_tester/count_reg [4]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[5] 
       (.C(clk),
        .CE(test_en),
        .D(p_0_in__1[5]),
        .Q(\u_tester/count_reg [5]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/count_reg[6] 
       (.C(clk),
        .CE(test_en),
        .D(p_0_in__1[6]),
        .Q(\u_tester/count_reg [6]),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/err_reg[0] 
       (.C(clk),
        .CE(test_stat[27]),
        .D(\err[0]_i_1_n_0 ),
        .Q(p_29_in[8]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/err_reg[1] 
       (.C(clk),
        .CE(test_stat[27]),
        .D(\err[1]_i_1_n_0 ),
        .Q(p_29_in[9]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/err_reg[2] 
       (.C(clk),
        .CE(test_stat[27]),
        .D(\err[2]_i_1_n_0 ),
        .Q(p_29_in[10]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/err_reg[3] 
       (.C(clk),
        .CE(test_stat[27]),
        .D(\err[3]_i_1_n_0 ),
        .Q(p_29_in[11]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/err_reg[4] 
       (.C(clk),
        .CE(test_stat[27]),
        .D(\err[4]_i_1_n_0 ),
        .Q(p_29_in[12]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/err_reg[5] 
       (.C(clk),
        .CE(test_stat[27]),
        .D(\err[5]_i_1_n_0 ),
        .Q(p_29_in[13]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/err_reg[6] 
       (.C(clk),
        .CE(test_stat[27]),
        .D(\err[6]_i_1_n_0 ),
        .Q(p_29_in[14]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/err_reg[7] 
       (.C(clk),
        .CE(test_stat[27]),
        .D(\err[7]_i_3_n_0 ),
        .Q(p_29_in[15]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/mask_count_reg[0] 
       (.C(clk),
        .CE(test_en),
        .D(test_en),
        .Q(\u_tester/mask_count_reg_n_0_[0] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[1] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/mask_count_reg_n_0_[0] ),
        .Q(\u_tester/mask_count_reg_n_0_[1] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[2] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/mask_count_reg_n_0_[1] ),
        .Q(\u_tester/mask_count_reg_n_0_[2] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[3] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/mask_count_reg_n_0_[2] ),
        .Q(\u_tester/mask_count_reg_n_0_[3] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[4] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/mask_count_reg_n_0_[3] ),
        .Q(\u_tester/mask_count_reg_n_0_[4] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[5] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/mask_count_reg_n_0_[4] ),
        .Q(\u_tester/mask_count_reg_n_0_[5] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[6] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/mask_count_reg_n_0_[5] ),
        .Q(\u_tester/mask_count_reg_n_0_[6] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/mask_count_reg[7] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/mask_count_reg_n_0_[6] ),
        .Q(\u_tester/mask_count_reg_n_0_[7] ),
        .R(\u_tester/mask_count ));
  FDRE \u_tester/rx_err_cnt_reg[0] 
       (.C(clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__2[0]),
        .Q(test_stat__0[0]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[1] 
       (.C(clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__2[1]),
        .Q(test_stat__0[1]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[2] 
       (.C(clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__2[2]),
        .Q(test_stat__0[2]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[3] 
       (.C(clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__2[3]),
        .Q(test_stat__0[3]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[4] 
       (.C(clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__2[4]),
        .Q(test_stat__0[4]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[5] 
       (.C(clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__2[5]),
        .Q(test_stat__0[5]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[6] 
       (.C(clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__2[6]),
        .Q(test_stat__0[6]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/rx_err_cnt_reg[7] 
       (.C(clk),
        .CE(\u_tester/rx_err_cnt0 ),
        .D(p_0_in__2[7]),
        .Q(test_stat__0[7]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/rx_err_s_reg 
       (.C(clk),
        .CE(1'b1),
        .D(rx_err_s_i_1_n_0),
        .Q(test_stat[21]),
        .R(1'b0));
  FDRE \u_tester/slip_cnt_reg[0] 
       (.C(clk),
        .CE(\slip_cnt[3]_i_2_n_0 ),
        .D(p_0_in__3[0]),
        .Q(test_stat__0[16]),
        .R(\u_tester/slip_cnt0 ));
  FDRE \u_tester/slip_cnt_reg[1] 
       (.C(clk),
        .CE(\slip_cnt[3]_i_2_n_0 ),
        .D(p_0_in__3[1]),
        .Q(test_stat__0[17]),
        .R(\u_tester/slip_cnt0 ));
  FDRE \u_tester/slip_cnt_reg[2] 
       (.C(clk),
        .CE(\slip_cnt[3]_i_2_n_0 ),
        .D(p_0_in__3[2]),
        .Q(test_stat__0[18]),
        .R(\u_tester/slip_cnt0 ));
  FDRE \u_tester/slip_cnt_reg[3] 
       (.C(clk),
        .CE(\slip_cnt[3]_i_2_n_0 ),
        .D(p_0_in__3[3]),
        .Q(test_stat__0[19]),
        .R(\u_tester/slip_cnt0 ));
  FDRE \u_tester/st_cur_reg[0] 
       (.C(clk),
        .CE(1'b1),
        .D(\st_cur[0]_i_1_n_0 ),
        .Q(\u_tester/st_cur [0]),
        .R(1'b0));
  FDRE \u_tester/st_cur_reg[1] 
       (.C(clk),
        .CE(1'b1),
        .D(\st_cur[1]_i_1_n_0 ),
        .Q(\u_tester/st_cur [1]),
        .R(1'b0));
  FDRE \u_tester/tx_cnt_reg[0] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[0]_i_2_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[0] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[10] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[8]_i_1_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[10] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[11] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[8]_i_1_n_4 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[11] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[12] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[12]_i_1_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[12] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[13] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[12]_i_1_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[13] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[14] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[12]_i_1_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[14] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[15] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[12]_i_1_n_4 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[15] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[16] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[16]_i_1_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[16] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[17] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[16]_i_1_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[17] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[18] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[16]_i_1_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[18] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[19] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[16]_i_1_n_4 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[19] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[1] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[0]_i_2_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[1] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[20] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[20]_i_1_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[20] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[21] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[20]_i_1_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[21] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[22] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[20]_i_1_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[22] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[23] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[20]_i_1_n_4 ),
        .Q(test_stat__0[8]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[24] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[24]_i_1_n_7 ),
        .Q(test_stat__0[9]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[25] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[24]_i_1_n_6 ),
        .Q(test_stat__0[10]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[26] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[24]_i_1_n_5 ),
        .Q(test_stat__0[11]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[27] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[24]_i_1_n_4 ),
        .Q(test_stat__0[12]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[28] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[28]_i_1_n_7 ),
        .Q(test_stat__0[13]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[29] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[28]_i_1_n_6 ),
        .Q(test_stat__0[14]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[2] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[0]_i_2_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[2] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[30] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[28]_i_1_n_5 ),
        .Q(test_stat__0[15]),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[3] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[0]_i_2_n_4 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[3] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[4] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[4]_i_1_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[4] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[5] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[4]_i_1_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[5] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[6] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[4]_i_1_n_5 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[6] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[7] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[4]_i_1_n_4 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[7] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[8] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[8]_i_1_n_7 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[8] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/tx_cnt_reg[9] 
       (.C(clk),
        .CE(\tx_cnt[0]_i_1_n_0 ),
        .D(\tx_cnt_reg[8]_i_1_n_6 ),
        .Q(\u_tester/tx_cnt_reg_n_0_[9] ),
        .R(\err[7]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[0] 
       (.C(clk),
        .CE(\u_tester/chk_en ),
        .D(\out[0]_i_1_n_0 ),
        .Q(\u_tester/chk_out [0]),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[1] 
       (.C(clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [0]),
        .Q(\u_tester/chk_out [1]),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[2] 
       (.C(clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [1]),
        .Q(\u_tester/chk_out [2]),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[3] 
       (.C(clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [2]),
        .Q(\u_tester/chk_out [3]),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[4] 
       (.C(clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [3]),
        .Q(\u_tester/chk_out [4]),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[5] 
       (.C(clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [4]),
        .Q(\u_tester/chk_out [5]),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[6] 
       (.C(clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [5]),
        .Q(\u_tester/chk_out [6]),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_chk/out_reg[7] 
       (.C(clk),
        .CE(\u_tester/chk_en ),
        .D(\u_tester/chk_out [6]),
        .Q(\u_tester/chk_out [7]),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[0] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/p_0_out ),
        .Q(\u_tester/u_gen/out_reg_n_0_[0] ),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[1] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/u_gen/out_reg_n_0_[0] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[1] ),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[2] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/u_gen/out_reg_n_0_[1] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[2] ),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[3] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/u_gen/out_reg_n_0_[2] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[3] ),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[4] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/u_gen/out_reg_n_0_[3] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[4] ),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[5] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/u_gen/out_reg_n_0_[4] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[5] ),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[6] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/u_gen/out_reg_n_0_[5] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[6] ),
        .R(\out[7]_i_1_n_0 ));
  FDRE \u_tester/u_gen/out_reg[7] 
       (.C(clk),
        .CE(test_en),
        .D(\u_tester/u_gen/out_reg_n_0_[6] ),
        .Q(\u_tester/u_gen/out_reg_n_0_[7] ),
        .R(\out[7]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair9" *) 
  LUT5 #(
    .INIT(32'h0808AA08)) 
    \w_outbus[0]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I1(\ctrl2_reg_n_0_[0] ),
        .I2(\dat_cache[25]_i_2_n_0 ),
        .I3(led_dnlink_on),
        .I4(\dat_cache[21]_i_2_n_0 ),
        .O(\w_outbus[0]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[10]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [10]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[10]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[10]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair69" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[10]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl2_reg_n_0_[10] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(ctrl0_rd[12]),
        .O(\w_outbus[10]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[11]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [11]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[11]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[11]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair74" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[11]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl2_reg_n_0_[11] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(\ctrl1_reg_n_0_[9] ),
        .O(\w_outbus[11]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[12]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [12]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[12]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[12]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair75" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[12]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl3_reg_n_0_[0] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(ctrl1_rd[10]),
        .O(\w_outbus[12]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[13]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [13]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[13]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[13]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair76" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[13]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl3_reg_n_0_[1] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(ctrl1_rd[11]),
        .O(\w_outbus[13]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[14]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [14]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[14]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[14]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair77" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[14]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl3_reg_n_0_[2] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(gain_attn2_local[0]),
        .O(\w_outbus[14]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[15]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [15]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[15]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[15]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair84" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[15]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl3_reg_n_0_[3] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(gain_attn2_local[1]),
        .O(\w_outbus[15]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[16]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [16]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[16]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[16]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair83" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[16]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl3_reg_n_0_[4] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(\ctrl1_reg_n_0_[0] ),
        .O(\w_outbus[16]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[17]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [17]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[17]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[17]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair82" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[17]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl3_reg_n_0_[5] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(\ctrl1_reg_n_0_[2] ),
        .O(\w_outbus[17]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[18]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [18]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[18]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[18]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair81" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[18]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl3_reg_n_0_[6] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(\ctrl1_reg_n_0_[3] ),
        .O(\w_outbus[18]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[19]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [19]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[19]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[19]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair66" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[19]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl3_reg_n_0_[7] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(\ctrl1_reg_n_0_[4] ),
        .O(\w_outbus[19]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[1]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [1]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[1]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[1]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair80" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[1]_i_2 
       (.I0(\dat_cache[21]_i_2_n_0 ),
        .I1(led_uplink_on),
        .I2(\dat_cache[25]_i_2_n_0 ),
        .I3(\ctrl2_reg_n_0_[1] ),
        .O(\w_outbus[1]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[20]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [20]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[20]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[20]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair79" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[20]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl3_reg_n_0_[8] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(rx_hisel),
        .O(\w_outbus[20]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[21]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [21]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[21]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[21]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair78" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[21]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(\ctrl3_reg_n_0_[9] ),
        .I2(\dat_cache[21]_i_2_n_0 ),
        .I3(tx_hisel),
        .O(\w_outbus[21]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[22]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [22]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[22]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[22]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair42" *) 
  LUT5 #(
    .INIT(32'h0BBBBBBB)) 
    \w_outbus[22]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(led_dnlink_on),
        .I2(dut_pgood),
        .I3(ctrl2_rd[0]),
        .I4(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .O(\w_outbus[22]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[23]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [23]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[23]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[23]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair43" *) 
  LUT5 #(
    .INIT(32'h0BBBBBBB)) 
    \w_outbus[23]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(led_uplink_on),
        .I2(dut_pgood),
        .I3(ctrl2_rd[1]),
        .I4(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .O(\w_outbus[23]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[24]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [24]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[24]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[24]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair44" *) 
  LUT5 #(
    .INIT(32'h0BBBBBBB)) 
    \w_outbus[24]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(led_error),
        .I2(dut_pgood),
        .I3(\ctrl2_reg_n_0_[2] ),
        .I4(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .O(\w_outbus[24]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[25]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [25]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[25]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[25]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair45" *) 
  LUT5 #(
    .INIT(32'h0BBBBBBB)) 
    \w_outbus[25]_i_2 
       (.I0(\dat_cache[25]_i_2_n_0 ),
        .I1(led_good),
        .I2(dut_pgood),
        .I3(\ctrl2_reg_n_0_[3] ),
        .I4(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .O(\w_outbus[25]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[26]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [26]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[26]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[26]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair46" *) 
  LUT5 #(
    .INIT(32'h0777FFFF)) 
    \w_outbus[26]_i_2 
       (.I0(\ctrl2_reg_n_0_[4] ),
        .I1(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .I2(\ctrl3_reg_n_0_[10] ),
        .I3(\rfmod_oe[8]_INST_0_i_1_n_0 ),
        .I4(dut_pgood),
        .O(\w_outbus[26]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[27]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [27]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[27]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[27]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair47" *) 
  LUT5 #(
    .INIT(32'h0777FFFF)) 
    \w_outbus[27]_i_2 
       (.I0(\ctrl2_reg_n_0_[5] ),
        .I1(\rfmod_out[14]_INST_0_i_3_n_0 ),
        .I2(\ctrl3_reg_n_0_[11] ),
        .I3(\rfmod_oe[8]_INST_0_i_1_n_0 ),
        .I4(dut_pgood),
        .O(\w_outbus[27]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'hF888F888F8888888)) 
    \w_outbus[28]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [28]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I3(\w_outbus[31]_i_4_n_0 ),
        .I4(sync),
        .I5(\ctrl2_reg_n_0_[12] ),
        .O(\w_outbus[28]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hF888F888F8888888)) 
    \w_outbus[29]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [29]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I3(\w_outbus[31]_i_4_n_0 ),
        .I4(\ctrl2_reg_n_0_[13] ),
        .I5(sync),
        .O(\w_outbus[29]_i_1_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[2]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [2]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[2]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[2]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair65" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[2]_i_2 
       (.I0(\dat_cache[21]_i_2_n_0 ),
        .I1(led_error),
        .I2(\dat_cache[25]_i_2_n_0 ),
        .I3(\ctrl2_reg_n_0_[2] ),
        .O(\w_outbus[2]_i_2_n_0 ));
  LUT6 #(
    .INIT(64'h88888888F8888888)) 
    \w_outbus[30]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [30]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I3(\ctrl2_reg_n_0_[14] ),
        .I4(\w_outbus[31]_i_4_n_0 ),
        .I5(sync),
        .O(\w_outbus[30]_i_1_n_0 ));
  LUT6 #(
    .INIT(64'hAAAAAEAAAAAAAAAA)) 
    \w_outbus[31]_i_1 
       (.I0(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[5] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[1] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[4] ),
        .I4(\w_outbus[31]_i_3_n_0 ),
        .I5(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .O(\u_spi_gpio_exp_master/w_outbus ));
  LUT6 #(
    .INIT(64'h88888888F8888888)) 
    \w_outbus[31]_i_2 
       (.I0(\u_spi_gpio_exp_master/in17 [31]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I3(\ctrl2_reg_n_0_[15] ),
        .I4(\w_outbus[31]_i_4_n_0 ),
        .I5(sync),
        .O(\w_outbus[31]_i_2_n_0 ));
  LUT4 #(
    .INIT(16'hFFEF)) 
    \w_outbus[31]_i_3 
       (.I0(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[2] ),
        .I1(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[3] ),
        .I2(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[0] ),
        .I3(\u_spi_gpio_exp_master/spi_bit_div_reg_n_0_[6] ),
        .O(\w_outbus[31]_i_3_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair58" *) 
  LUT5 #(
    .INIT(32'h00020200)) 
    \w_outbus[31]_i_4 
       (.I0(dut_pgood),
        .I1(rfmod_id[3]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[0]),
        .I4(rfmod_id[1]),
        .O(\w_outbus[31]_i_4_n_0 ));
  LUT4 #(
    .INIT(16'h8F88)) 
    \w_outbus[3]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [3]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\w_outbus[3]_i_2_n_0 ),
        .I3(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .O(\w_outbus[3]_i_1_n_0 ));
  (* SOFT_HLUTNM = "soft_lutpair71" *) 
  LUT4 #(
    .INIT(16'hB0BB)) 
    \w_outbus[3]_i_2 
       (.I0(\dat_cache[21]_i_2_n_0 ),
        .I1(led_good),
        .I2(\dat_cache[25]_i_2_n_0 ),
        .I3(\ctrl2_reg_n_0_[3] ),
        .O(\w_outbus[3]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'h8888F888)) 
    \w_outbus[4]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [4]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I3(\ctrl2_reg_n_0_[4] ),
        .I4(\dat_cache[25]_i_2_n_0 ),
        .O(\w_outbus[4]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h8888F888)) 
    \w_outbus[5]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [5]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I3(\ctrl2_reg_n_0_[5] ),
        .I4(\dat_cache[25]_i_2_n_0 ),
        .O(\w_outbus[5]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h8888F888)) 
    \w_outbus[6]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [6]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I3(\ctrl2_reg_n_0_[6] ),
        .I4(\dat_cache[25]_i_2_n_0 ),
        .O(\w_outbus[6]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h8888F888)) 
    \w_outbus[7]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [7]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I3(\ctrl2_reg_n_0_[7] ),
        .I4(\dat_cache[25]_i_2_n_0 ),
        .O(\w_outbus[7]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h8888F888)) 
    \w_outbus[8]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [8]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I3(\ctrl2_reg_n_0_[8] ),
        .I4(\dat_cache[25]_i_2_n_0 ),
        .O(\w_outbus[8]_i_1_n_0 ));
  LUT5 #(
    .INIT(32'h8888F888)) 
    \w_outbus[9]_i_1 
       (.I0(\u_spi_gpio_exp_master/in17 [9]),
        .I1(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[2] ),
        .I2(\u_spi_gpio_exp_master/FSM_onehot_state_reg_n_0_[0] ),
        .I3(\ctrl2_reg_n_0_[9] ),
        .I4(\dat_cache[25]_i_2_n_0 ),
        .O(\w_outbus[9]_i_1_n_0 ));
  LUT2 #(
    .INIT(4'h6)) 
    \wr_addr_reg[0]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [0]),
        .I1(mem_reg_0_3_0_5_i_1__0_n_0),
        .O(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_next [0]));
  LUT3 #(
    .INIT(8'hD2)) 
    \wr_addr_reg[0]_i_1__0 
       (.I0(\u_i2c_master/u_i2c_master/data_out_valid0 ),
        .I1(\wr_addr_reg[1]_i_2_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [0]),
        .O(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_next [0]));
  (* SOFT_HLUTNM = "soft_lutpair89" *) 
  LUT3 #(
    .INIT(8'h6A)) 
    \wr_addr_reg[1]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [1]),
        .I1(mem_reg_0_3_0_5_i_1__0_n_0),
        .I2(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [0]),
        .O(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_next [1]));
  (* SOFT_HLUTNM = "soft_lutpair61" *) 
  LUT4 #(
    .INIT(16'hD2F0)) 
    \wr_addr_reg[1]_i_1__0 
       (.I0(\u_i2c_master/u_i2c_master/data_out_valid0 ),
        .I1(\wr_addr_reg[1]_i_2_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [1]),
        .I3(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [0]),
        .O(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_next [1]));
  LUT6 #(
    .INIT(64'h0000900990090000)) 
    \wr_addr_reg[1]_i_2 
       (.I0(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [1]),
        .I1(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[1] ),
        .I2(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [0]),
        .I3(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[0] ),
        .I4(\u_i2c_master/u_i2c_master/u_rdata_fifo/rd_ptr_reg_reg_n_0_[2] ),
        .I5(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [2]),
        .O(\wr_addr_reg[1]_i_2_n_0 ));
  LUT5 #(
    .INIT(32'h00008000)) 
    wr_cache_reg_0_7_0_5_i_1
       (.I0(rfmod_id[3]),
        .I1(rfmod_id[0]),
        .I2(rfmod_id[2]),
        .I3(rfmod_id[1]),
        .I4(i2c_cache_addr[2]),
        .O(wr_cache_reg_0_7_0_5_i_1_n_0));
  LUT6 #(
    .INIT(64'h0000103000001000)) 
    wr_cache_reg_0_7_0_5_i_2
       (.I0(led_good),
        .I1(i2c_cache_addr[2]),
        .I2(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I3(i2c_cache_addr[0]),
        .I4(i2c_cache_addr[1]),
        .I5(\ctrl0_reg_n_0_[13] ),
        .O(wr_cache_reg_0_7_0_5_i_2_n_0));
  LUT6 #(
    .INIT(64'hAFAEAAAAAAAEAAAA)) 
    wr_cache_reg_0_7_0_5_i_3
       (.I0(wr_cache_reg_0_7_0_5_i_5_n_0),
        .I1(led_dnlink_on),
        .I2(i2c_cache_addr[1]),
        .I3(i2c_cache_addr[0]),
        .I4(wr_cache_reg_0_7_0_5_i_1_n_0),
        .I5(led_error),
        .O(wr_cache_reg_0_7_0_5_i_3_n_0));
  LUT6 #(
    .INIT(64'h101010F010101010)) 
    wr_cache_reg_0_7_0_5_i_4
       (.I0(wr_cache_reg_0_7_0_5_i_6_n_0),
        .I1(\ctrl0_reg_n_0_[13] ),
        .I2(wr_cache_reg_0_7_0_5_i_1_n_0),
        .I3(i2c_cache_addr[0]),
        .I4(i2c_cache_addr[1]),
        .I5(led_uplink_on),
        .O(wr_cache_reg_0_7_0_5_i_4_n_0));
  LUT6 #(
    .INIT(64'h000000000000A820)) 
    wr_cache_reg_0_7_0_5_i_5
       (.I0(i2c_cache_addr[1]),
        .I1(i2c_cache_addr[0]),
        .I2(wr_cache_reg_0_7_0_5_i_7_n_0),
        .I3(wr_cache_reg_0_7_0_5_i_8_n_0),
        .I4(i2c_cache_addr[2]),
        .I5(\ctrl0_reg_n_0_[13] ),
        .O(wr_cache_reg_0_7_0_5_i_5_n_0));
  LUT6 #(
    .INIT(64'h000FFFFF1111FFFF)) 
    wr_cache_reg_0_7_0_5_i_6
       (.I0(tx_active[0]),
        .I1(p_0_in0_in),
        .I2(tx_active[1]),
        .I3(\ctrl1_reg_n_0_[12] ),
        .I4(i2c_cache_addr[1]),
        .I5(i2c_cache_addr[0]),
        .O(wr_cache_reg_0_7_0_5_i_6_n_0));
  (* SOFT_HLUTNM = "soft_lutpair8" *) 
  LUT5 #(
    .INIT(32'h80000000)) 
    wr_cache_reg_0_7_0_5_i_7
       (.I0(\ctrl0_reg_n_0_[0] ),
        .I1(rfmod_id[3]),
        .I2(rfmod_id[0]),
        .I3(rfmod_id[2]),
        .I4(rfmod_id[1]),
        .O(wr_cache_reg_0_7_0_5_i_7_n_0));
  (* SOFT_HLUTNM = "soft_lutpair3" *) 
  LUT5 #(
    .INIT(32'h80000000)) 
    wr_cache_reg_0_7_0_5_i_8
       (.I0(fdd_en_b),
        .I1(rfmod_id[3]),
        .I2(rfmod_id[0]),
        .I3(rfmod_id[2]),
        .I4(rfmod_id[1]),
        .O(wr_cache_reg_0_7_0_5_i_8_n_0));
  LUT4 #(
    .INIT(16'h0004)) 
    wr_cache_reg_0_7_6_7_i_1
       (.I0(\ctrl0_reg_n_0_[13] ),
        .I1(\rfmod_oe[13]_INST_0_i_2_n_0 ),
        .I2(i2c_cache_addr[2]),
        .I3(wr_cache_reg_0_7_6_7_i_2_n_0),
        .O(wr_cache_reg_0_7_6_7_i_1_n_0));
  LUT6 #(
    .INIT(64'hF0DDFFFFFFDDFFFF)) 
    wr_cache_reg_0_7_6_7_i_2
       (.I0(\rfmod_out[9]_INST_0_i_2_n_0 ),
        .I1(\ctrl0_reg_n_0_[0] ),
        .I2(fdd_en_b),
        .I3(i2c_cache_addr[0]),
        .I4(i2c_cache_addr[1]),
        .I5(wr_cache_reg_0_7_6_7_i_3_n_0),
        .O(wr_cache_reg_0_7_6_7_i_2_n_0));
  (* SOFT_HLUTNM = "soft_lutpair138" *) 
  LUT2 #(
    .INIT(4'h1)) 
    wr_cache_reg_0_7_6_7_i_3
       (.I0(tx_active[1]),
        .I1(\ctrl1_reg_n_0_[12] ),
        .O(wr_cache_reg_0_7_6_7_i_3_n_0));
  (* SOFT_HLUTNM = "soft_lutpair89" *) 
  LUT4 #(
    .INIT(16'h6AAA)) 
    \wr_ptr_reg[2]_i_1 
       (.I0(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [2]),
        .I1(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [0]),
        .I2(mem_reg_0_3_0_5_i_1__0_n_0),
        .I3(\u_i2c_master/u_i2c_master/u_wdata_fifo/wr_ptr_reg_reg [1]),
        .O(wr_ptr_next));
  (* SOFT_HLUTNM = "soft_lutpair61" *) 
  LUT5 #(
    .INIT(32'hD2F0F0F0)) 
    \wr_ptr_reg[2]_i_1__0 
       (.I0(\u_i2c_master/u_i2c_master/data_out_valid0 ),
        .I1(\wr_addr_reg[1]_i_2_n_0 ),
        .I2(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [2]),
        .I3(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [0]),
        .I4(\u_i2c_master/u_i2c_master/u_rdata_fifo/wr_ptr_reg_reg [1]),
        .O(wr_ptr_next__0));
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
