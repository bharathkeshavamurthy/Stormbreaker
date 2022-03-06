########################################################################
## LMS7 DIQ interface
########################################################################

#set for the maximum 2x clock rate of 80 MHz (40 Msps)
#using the main rx clock for both rx input signals and tx output signals
create_clock -period 12.5 -name lms_diq2_mclk [get_ports lms_diq2_mclk]

#name internal clocks for contraining the DIQ IOs
create_generated_clock -name adc_clk [get_pins "u_trxiq_top/u_clocks/mmcm_mclk/CLKOUT0"]
create_generated_clock -name dac_fclk [get_pins "u_trxiq_top/u_clocks/mmcm_mclk/CLKOUT1"]

#pick the mmcm locked path for FCLK delay generation with the BUFGMUX
set_case_analysis 1 [get_pins "u_trxiq_top/u_clocks/clkmux_*/S"]

#false path for the sampling of the mclk when its very slow (unlocked)
set_false_path -from [get_ports lms_diq2_mclk] -to [get_pins {u_trxiq_top/u_clocks/*/D}]

#the sys clock samples the mclk for very slow rates below MMCM capabilities
#these delays do not matter, they are simply to make the tools accept it
set_input_delay -clock [get_clocks clk_fpga_0] -min 0 [get_ports lms_diq2_mclk]
set_input_delay -clock [get_clocks clk_fpga_0] -max 10 [get_ports lms_diq2_mclk]

#The mclk edges are aligned with the data transitions using an intentional delay in the lms7002m
#Use 1ns for the maximum/worst setup time as specified by the lms700m data sheet.
#Use 0.1ns + quarter cycle duration for the minimum/worst hold time specification.
set_input_delay -clock [get_clocks adc_clk] -max 1 [get_ports {lms_diq2_iqseldir lms_diq2_d[*]}]
set_input_delay -clock [get_clocks adc_clk] -max 1 [get_ports {lms_diq2_iqseldir lms_diq2_d[*]}] -clock_fall -add_delay
set_input_delay -clock [get_clocks adc_clk] -min 4.046 [get_ports {lms_diq2_iqseldir lms_diq2_d[*]}]
set_input_delay -clock [get_clocks adc_clk] -min 4.046 [get_ports {lms_diq2_iqseldir lms_diq2_d[*]}] -clock_fall -add_delay

set_output_delay -clock [get_clocks adc_clk] -max -3.646 [get_ports {lms_diq1_iqseldir lms_diq1_d[*]}]
set_output_delay -clock [get_clocks adc_clk] -max -3.646 [get_ports {lms_diq1_iqseldir lms_diq1_d[*]}] -clock_fall -add_delay
set_output_delay -clock [get_clocks adc_clk] -min 0 [get_ports {lms_diq1_iqseldir lms_diq1_d[*]}]
set_output_delay -clock [get_clocks adc_clk] -min 0 [get_ports {lms_diq1_iqseldir lms_diq1_d[*]}] -clock_fall -add_delay

set_output_delay -clock [get_clocks dac_fclk] -max -3.646 [get_ports {lms_diq1_fclk}]
set_output_delay -clock [get_clocks dac_fclk] -max -3.646 [get_ports {lms_diq1_fclk}] -clock_fall -add_delay
set_output_delay -clock [get_clocks dac_fclk] -min 0 [get_ports {lms_diq1_fclk}]
set_output_delay -clock [get_clocks dac_fclk] -min 0 [get_ports {lms_diq1_fclk}] -clock_fall -add_delay

########################################################################
## RFMOD GPIOs
########################################################################
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {rfmod_gpio[*]}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {rfmod_gpio[*]}]

########################################################################
## GPIO
########################################################################
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {clkbuff_lock}]
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {clkbuff_irq}]
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {brd_rev[*]}]
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {fpga_red_led}]
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {pmod_3}]

set_false_path -from [get_ports {clkbuff_lock}]
set_false_path -from [get_ports {clkbuff_irq}]
set_false_path -from [get_ports {brd_rev[*]}]
set_false_path -from [get_ports {fpga_red_led}]
set_false_path -from [get_ports {pmod_3}]

set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {pmod_2}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {pmod_3}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {pmod_7}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {lms_dig_rst}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {lms_hw_rxen}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {lms_hw_txen}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {clkbuff_rst_n}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {fpga_red_led}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {fpga_grn_led_n}]

set_false_path -to [get_ports {pmod_2}]
set_false_path -to [get_ports {pmod_3}]
set_false_path -to [get_ports {pmod_7}]
set_false_path -to [get_ports {lms_dig_rst}]
set_false_path -to [get_ports {lms_hw_rxen}]
set_false_path -to [get_ports {lms_hw_txen}]
set_false_path -to [get_ports {clkbuff_rst_n}]
set_false_path -to [get_ports {fpga_red_led}]
set_false_path -to [get_ports {fpga_grn_led_n}]

set_output_delay -clock [get_clocks clk_fpga_0] -max -3.646 [get_ports {lms_spi_*}]
set_output_delay -clock [get_clocks clk_fpga_0] -max -3.646 [get_ports {lms_spi_*}] -clock_fall -add_delay
set_output_delay -clock [get_clocks clk_fpga_0] -min 0 [get_ports {lms_spi_*}]
set_output_delay -clock [get_clocks clk_fpga_0] -min 0 [get_ports {lms_spi_*}] -clock_fall -add_delay
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {lms_spi_sdo}]

#uplink interconnect IOs
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_up_psnt_n_in}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_up_sclk}]
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_up_sclk}]
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_up_is_head_n}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_up_is_head_n}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_up_psnt_n_out}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_up_sda}]
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_up_sda}]

set_false_path -from [get_ports {icnt_up_psnt_n_in}]
set_false_path -to [get_ports {icnt_up_sclk}]
set_false_path -from [get_ports {icnt_up_sclk}]
set_false_path -from [get_ports {icnt_up_is_head_n}]
set_false_path -to [get_ports {icnt_up_is_head_n}]
set_false_path -to [get_ports {icnt_up_psnt_n_out}]
set_false_path -to [get_ports {icnt_up_sda}]
set_false_path -from [get_ports {icnt_up_sda}]

#downlink interconnect IOs
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_dn_psnt_n_in}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_dn_sda}]
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_dn_sda}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_dn_psnt_n_out}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_dn_sclk}]
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_dn_sclk}]
set_output_delay -clock [get_clocks adc_clk] 0 [get_ports {icnt_dn_trigger}]
set_input_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_dn_is_tail_n}]
set_output_delay -clock [get_clocks clk_fpga_0] 0 [get_ports {icnt_dn_is_tail_n}]

set_false_path -from [get_ports {icnt_dn_psnt_n_in}]
set_false_path -to [get_ports {icnt_dn_sda}]
set_false_path -from [get_ports {icnt_dn_sda}]
set_false_path -to [get_ports {icnt_dn_psnt_n_out}]
set_false_path -to [get_ports {icnt_dn_sclk}]
set_false_path -from [get_ports {icnt_dn_sclk}]
set_false_path -to [get_ports {icnt_dn_trigger}]
set_false_path -from [get_ports {icnt_dn_is_tail_n}]
set_false_path -to [get_ports {icnt_dn_is_tail_n}]
