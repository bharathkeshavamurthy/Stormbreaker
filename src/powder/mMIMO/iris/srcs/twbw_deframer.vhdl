------------------------------------------------------------------------
-- TWBW Deframer - DAC sample deframer with time and burst controls
--
-- Input port documentation:
--
-- The input port contains framed sample data with 4 header lines.
-- The last transaction in each frame has a tlast asserted.
-- The input port data width may be larger than 32 bits,
-- but only the first 32 bits of each header line are used.
--
-- Use the continuous flag to indicate additional contiguous packets.
-- End a burst by not specifying continuous flag on the last packet.
--
-- A time stamp may be specified on every packet,
-- but it is only waited on for the first packet in a burst,
-- and on subsequent packets proceeding an error event.
--
-- Input bus format:
--   in0[31] - time stamp flag indicates valid time
--   in0[27] - continuous flag (1 = more data, 0 = end burst)
--   in0[26] - wait for trigger flag
--   in0[25] - force a status message with sequence
--   in0[23:16] - identifying tag (forwarded to status port)
--   in1[31:16] - 16-bit sequence value to write back in status
--   in1[15:0] - number of data transfers counting down to zero
--   in2[31:0] - 64 bit time high in clock ticks (used with time flag)
--   in3[31:0] - 64 bit time low in clock ticks (used with time flag)
--
-- Status port documentation:
--
-- Each status bus message consists of four 32-bit transactions.
-- The last 32-bit transaction in each message has a tlast asserted.
-- The message contains flags, and a time.
-- The tag is forwarded to the status port to help identify the message.
--
-- Status bus format:
--   stat0[31] - time stamp flag
--   stat0[30] - time error event occurred
--   stat0[29] - underflow event occurred
--   stat0[28] - burst end event occurred
--   stat0[25] - sequence flag encountered
--   stat0[24] - sequence error detected
--   stat0[23:16] - identifying tag (forwarded to status port)
--   stat1[31:16] - 16-bit sequence (forwarded to status port)
--   stat2[31:0] - 64 bit time high in clock ticks (used with time flag)
--   stat3[31:0] - 64 bit time low in clock ticks (used with time flag)
--
-- Copyright (c) 2015-2015 Lime Microsystems
-- Copyright (c) 2015-2015 Andrew "bunnie" Huang
-- SPDX-License-Identifier: Apache-2.0
-- http://www.apache.org/licenses/LICENSE-2.0
------------------------------------------------------------------------

library work;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity twbw_deframer is
    generic(
        DATA_WIDTH : natural := 64;
        TIME_WIDTH : natural := 64);
    port(
        -- The DAC clock domain used for all interfaces.
        clk : in std_logic;

        -- synchronous reset
        rst : in std_logic;

        -- an external trigger used in trigger wait mode
        in_trigger : in std_logic;

        -- The current time in clock ticks.
        in_time : in unsigned(TIME_WIDTH-1 downto 0);

        -- The output DAC interface:
        -- There is no valid signal, a ready signal with no data => underflow.
        -- To allow for frame overhead, this bus cannot be ready every cycle.
        -- Many interfaces do not consume transmit data at every clock cycle.
        -- If this is not the case, we recommend doubling the DAC data width.
        dac_tdata : out std_logic_vector(DATA_WIDTH-1 downto 0);
        dac_tvalid : out std_logic;
        dac_tready : in std_logic;

        -- Input stream interface:
        -- The tuser signal indicates metadata and not sample data
        in_tdata : in std_logic_vector(DATA_WIDTH-1 downto 0);
        in_tuser : in std_logic_vector(0 downto 0);
        in_tlast : in std_logic;
        in_tvalid : in std_logic;
        in_tready : out std_logic;

        -- status bus interface
        stat_tdata : out std_logic_vector(127 downto 0);
        stat_tlast : out std_logic;
        stat_tvalid : out std_logic;
        stat_tready : in std_logic;

        -- Transmit activity indicator based on internal state.
        -- Example: use this signal to drive external switches and LEDs.
        dac_active : out std_logic
    );
end entity twbw_deframer;

architecture rtl of twbw_deframer is

    -- state machine inspects the DAC from these signals
    constant dac_tdata_zeros : std_logic_vector(DATA_WIDTH-1 downto 0) := (others => '0');
    signal dac_fifo_out_valid : std_logic;
    signal dac_fifo_out_ready : std_logic;
    signal dac_fifo_out_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal dac_fifo_in_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal dac_fifo_in_valid : std_logic;
    signal dac_fifo_in_ready : std_logic;
    signal dac_fifo_out_trigger : boolean;

    -- state machine drives the status messages with these signals
    signal stat_tdata_word0 : std_logic_vector(31 downto 0) := (others => '0');
    signal stat_tdata_word1 : std_logic_vector(31 downto 0) := (others => '0');
    signal stat_tdata_time64 : std_logic_vector(63 downto 0) := (others => '0');
    signal stat_fifo_in_valid : std_logic;
    signal stat_fifo_in_ready : std_logic;

    -- state machine controls the outgoing framed bus signals
    signal framed_fifo_in_full : std_logic_vector(2+DATA_WIDTH-1 downto 0);
    signal framed_fifo_out_full : std_logic_vector(2+DATA_WIDTH-1 downto 0);
    signal framed_fifo_out_data : std_logic_vector(DATA_WIDTH-1 downto 0);
    signal framed_fifo_out_user : std_logic_vector(0 downto 0);
    signal framed_fifo_out_last : std_logic;
    signal framed_fifo_out_valid : std_logic;
    signal framed_fifo_out_ready : std_logic;

    -- internal dac active indicator
    signal dac_active_i : boolean;
    signal dac_active_or_trigger : boolean;

    -- state machine enumerations
    type state_type is (
        STATE_TX_IDLE,
        STATE_HDR0_IN,
        STATE_HDR1_IN,
        STATE_HDR2_IN,
        STATE_HDR3_IN,
        STATE_TRICKLE,
        STATE_WAIT_TIME,
        STATE_SAMPS_IN,
        STATE_DRAIN_SAMPS,
        STATE_STAT_OUT);
    signal state : state_type;
    signal state_num : unsigned(7 downto 0);

    -- state variables set by the state machine
    signal time_flag : std_logic := '0';
    signal continuous_flag : std_logic := '0';
    signal trigger_flag : std_logic := '0';
    signal seq_flag : std_logic := '0';
    signal stat_tag : std_logic_vector(7 downto 0) := (others => '0');
    signal stat_seq : unsigned(15 downto 0) := to_unsigned(0, 16);
    signal stat_seq_next : unsigned(15 downto 0);
    signal stat_seq_expected : unsigned(15 downto 0);
    signal stream_time : unsigned(TIME_WIDTH-1 downto 0) := to_unsigned(0, TIME_WIDTH);
    signal underflow : std_logic := '0';
    signal time_error : std_logic := '0';
    signal seq_error : std_logic := '0';
    signal end_burst : std_logic := '0';
    signal trickle : std_logic_vector(3 downto 0) := "0000";

    --ila debug
    signal CONTROL_ILA : std_logic_vector(35 downto 0);
    signal DATA_ILA : std_logic_vector(63 downto 0);
    signal TRIG_ILA : std_logic_vector(7 downto 0);

begin
    assert (TIME_WIDTH <= 64) report "twbw_deframer: time width too large" severity failure;
    assert (DATA_WIDTH >= 32) report "twbw_deframer: data width too small" severity failure;

    state_num <= to_unsigned(state_type'pos(state), 8);

    --boolean condition tracking
    dac_active_or_trigger <= dac_active_i or (state = STATE_WAIT_TIME and dac_fifo_out_trigger);
    dac_active <= dac_fifo_out_valid when dac_active_or_trigger else '0';

    --------------------------------------------------------------------
    -- short fifo between dac in and state machine
    -- this fifo ensures a continuous streaming despite deframing overhead
    --------------------------------------------------------------------
    dac_fifo: entity work.StreamFifoSrl32
    port map (
        clk => clk,
        rst => rst,
        in_data => dac_fifo_in_data,
        in_valid => dac_fifo_in_valid,
        in_ready => dac_fifo_in_ready,
        out_data => dac_fifo_out_data,
        out_valid => dac_fifo_out_valid,
        out_ready => dac_fifo_out_ready
    );

    dac_tdata <= dac_fifo_out_data when (dac_fifo_out_valid = '1' and dac_active_or_trigger) else dac_tdata_zeros;
    dac_fifo_in_data <= framed_fifo_out_data;
    dac_fifo_in_valid <= framed_fifo_out_valid when (state = STATE_SAMPS_IN or state = STATE_TRICKLE) else '0';

    dac_tvalid <= dac_fifo_out_valid when dac_active_or_trigger else '0';
    dac_fifo_out_ready <=
        dac_tready when dac_active_or_trigger else --output exactly on triggered cycle
        '0' when (state = STATE_TRICKLE) else --plug the fifo when waiting
        '0' when (state = STATE_WAIT_TIME) else --plug the fifo when waiting
        '1' when not dac_active_i else --always draining the fifo when inactive
        dac_tready; --otherwise following the output strobe condition

    --hold previous cycle until input valid occurs
    --this makes the trigger condition sticky
    trigger_work: process (clk, trigger_flag, in_trigger, time_flag, in_time, stream_time)
        variable this_cycle_trigger : boolean;
        variable prev_cycle_trigger : boolean;
    begin

        --trigger conditions
        this_cycle_trigger :=
            (trigger_flag = '1' and in_trigger = '1') or
            (time_flag = '1' and in_time = stream_time);

        --was this cycle or the previous triggered?
        dac_fifo_out_trigger <= this_cycle_trigger or prev_cycle_trigger;

        --sticky bit logic for triggered
        if (rising_edge(clk)) then
            if (dac_tready = '1') then
                prev_cycle_trigger := false;
            else
                prev_cycle_trigger := this_cycle_trigger;
            end if;
        end if;

    end process trigger_work;

    --------------------------------------------------------------------
    -- short fifo between state machine and stat bus
    -- this fifo changes size and gives storage space
    --------------------------------------------------------------------
    --stat_fifo: entity work.stat_msg_fifo128
    --port map (
        --clk => clk,
        --rst => rst,
        --in_tdata => stat_fifo_in_data,
        --in_tvalid => stat_fifo_in_valid,
        --in_tready => stat_fifo_in_ready,
        --out_tdata => stat_tdata,
        --out_tlast => stat_tlast,
        --out_tvalid => stat_tvalid,
        --out_tready => stat_tready
    --);
    --least significant word32 packing
    stat_tdata <=
        stat_tdata_time64(31 downto 0) &
        stat_tdata_time64(63 downto 32) &
        stat_tdata_word1 &
        stat_tdata_word0;
    stat_tlast <= '1';
    stat_tvalid <= stat_fifo_in_valid;
    stat_fifo_in_ready <= stat_tready;

    stat_fifo_in_valid <= '1' when (state = STATE_STAT_OUT)
        --otherwise force out a message on the final transfer when a sequence packet is requested
        --do not check backpressure ready signal, sequence packets can drop if there is no space
        else (framed_fifo_out_valid and framed_fifo_out_ready and framed_fifo_out_last and (seq_flag or seq_error));
    stat_tdata_time64(TIME_WIDTH-1 downto 0) <= std_logic_vector(in_time);
    stat_tdata_word1(31 downto 16) <= std_logic_vector(stat_seq);
    stat_tdata_word0(31) <= '1'; --has a valid time stamp
    stat_tdata_word0(30) <= time_error; --time error
    stat_tdata_word0(29) <= underflow; --underflow
    stat_tdata_word0(28) <= end_burst; --burst end
    stat_tdata_word0(27) <= '0';
    stat_tdata_word0(26) <= '0';
    stat_tdata_word0(25) <= seq_flag; --sequence
    stat_tdata_word0(24) <= seq_error;
    stat_tdata_word0(23 downto 16) <= stat_tag; --tag from input
    stat_tdata_word0(15 downto 0) <= (others => '0');

    --------------------------------------------------------------------
    -- fifo between input stream and state machine
    --------------------------------------------------------------------
    framed_fifo : entity work.StreamFifoSrl32
    port map (
        clk => clk,
        rst => rst,
        in_data => framed_fifo_in_full,
        in_valid => in_tvalid,
        in_ready => in_tready,
        out_data => framed_fifo_out_full,
        out_valid => framed_fifo_out_valid,
        out_ready => framed_fifo_out_ready
    );

    framed_fifo_out_ready <= '1' when (
        state = STATE_HDR0_IN or
        state = STATE_HDR1_IN or
        state = STATE_HDR2_IN or
        state = STATE_HDR3_IN or
        state = STATE_DRAIN_SAMPS) else dac_fifo_in_ready when (state = STATE_SAMPS_IN or state = STATE_TRICKLE) else '0';

    framed_fifo_out_data <= framed_fifo_out_full(DATA_WIDTH-1 downto 0);
    framed_fifo_out_last <= framed_fifo_out_full(DATA_WIDTH);
    framed_fifo_out_user <= framed_fifo_out_full(DATA_WIDTH+1 downto DATA_WIDTH+1);

    framed_fifo_in_full(DATA_WIDTH-1 downto 0) <= in_tdata;
    framed_fifo_in_full(DATA_WIDTH) <= in_tlast;
    framed_fifo_in_full(DATA_WIDTH+1 downto DATA_WIDTH+1) <= in_tuser;

    stat_seq_expected <= stat_seq + 1;
    stat_seq_next <= unsigned(framed_fifo_out_data(31 downto 16));

    --------------------------------------------------------------------
    -- deframer state machine
    --------------------------------------------------------------------
    process (clk)
        variable after_wait_state : state_type;
    begin

    if (rising_edge(clk)) then

        --underflow indicator is sticky, only possible when dac is in active state
        if (rst = '1' or not dac_active_i) then
            underflow <= '0';
        elsif (state = STATE_TX_IDLE and dac_fifo_out_valid = '0') then
            underflow <= '0';
        else
            underflow <= underflow or (dac_tready and not dac_fifo_out_valid);
        end if;

        if (rst = '1') then
            dac_active_i <= false;
            state <= STATE_TX_IDLE;
            time_flag <= '0';
            continuous_flag <= '0';
            trigger_flag <= '0';
            seq_flag <= '0';
            stat_tag <= (others => '0');
            stat_seq <= to_unsigned(0, 16);
            stream_time <= to_unsigned(0, TIME_WIDTH);
            time_error <= '0';
            seq_error <= '0';
            end_burst <= '0';
        else case state is

        when STATE_TX_IDLE =>
            --let the dac fifo drain its contents from a previous state
            --before the idle values are actually restored
            if (dac_fifo_out_valid = '0') then
                state <= STATE_HDR0_IN;
                dac_active_i <= false;
                time_error <= '0';
                seq_error <= '0';
                end_burst <= '0';
            end if;

        when STATE_HDR0_IN =>
            if (framed_fifo_out_valid = '1' and framed_fifo_out_ready = '1') then
                time_flag <= framed_fifo_out_data(31);
                continuous_flag <= framed_fifo_out_data(27);
                trigger_flag <= framed_fifo_out_data(26);
                seq_flag <= framed_fifo_out_data(25);
                stat_tag <= framed_fifo_out_data(23 downto 16);
                seq_error <= '0'; --clear
                if (framed_fifo_out_last = '1') then
                    state <= STATE_TX_IDLE; --tlast here?
                else
                    state <= STATE_HDR1_IN;
                end if;
            elsif (underflow = '1') then
                state <= STATE_STAT_OUT;
            end if;

        when STATE_HDR1_IN =>
            if (framed_fifo_out_valid = '1' and framed_fifo_out_ready = '1') then
                --requires dac active and sequence mismatch, no error for start of burst
                if (dac_active_i and stat_seq_next /= stat_seq_expected) then
                    seq_error <= '1';
                end if;
                stat_seq <= stat_seq_next;
                if (framed_fifo_out_last = '1') then
                    state <= STATE_TX_IDLE; --tlast here?
                else
                    state <= STATE_HDR2_IN;
                end if;
            end if;

        when STATE_HDR2_IN =>
            if (framed_fifo_out_valid = '1' and framed_fifo_out_ready = '1') then
                stream_time(TIME_WIDTH-1 downto 32) <= unsigned(framed_fifo_out_data(TIME_WIDTH-32-1 downto 0));
                if (framed_fifo_out_last = '1') then
                    state <= STATE_TX_IDLE; --tlast here?
                else
                    state <= STATE_HDR3_IN;
                end if;
            end if;

        when STATE_HDR3_IN =>
            if (framed_fifo_out_valid = '1' and framed_fifo_out_ready = '1') then
                stream_time(31 downto 0) <= unsigned(framed_fifo_out_data(31 downto 0));
                if (framed_fifo_out_last = '1') then
                    state <= STATE_TX_IDLE; --tlast here?
                elsif (underflow = '1') then --underflow while reading header
                    state <= STATE_DRAIN_SAMPS;
                elsif (dac_active_i) then --dont wait, keep the packet flowing
                    state <= STATE_SAMPS_IN;
                else
                    state <= STATE_TRICKLE;
                    trickle <= "0000";
                end if;
            end if;

        when STATE_TRICKLE =>
            if (framed_fifo_out_valid = '1' and framed_fifo_out_ready = '1') then
                trickle <= trickle(2 downto 0) & '1';
                if (framed_fifo_out_last = '1') then
                    state <= STATE_WAIT_TIME;
                    if (continuous_flag = '1') then
                        after_wait_state := STATE_HDR0_IN;
                    else
                        after_wait_state := STATE_STAT_OUT;
                        end_burst <= '1';
                    end if;
                elsif (trickle = "1111") then
                    state <= STATE_WAIT_TIME;
                    after_wait_state := STATE_SAMPS_IN;
                end if;
            end if;

        when STATE_WAIT_TIME =>

            --wait for a sample passing cycle
            if (dac_tready = '0' or dac_fifo_out_valid = '0') then
                --pass

            --no condition specified, go to samples in
            elsif (time_flag = '0' and trigger_flag = '0') then
                dac_active_i <= true;
                state <= after_wait_state;

            --trigger flag specified and triggered
            elsif (trigger_flag = '1' and dac_fifo_out_trigger) then
                dac_active_i <= true;
                state <= after_wait_state;

            --time flag specified and time matched
            elsif (time_flag = '1' and dac_fifo_out_trigger) then
                dac_active_i <= true;
                state <= after_wait_state;

            --the requested time has expired!
            elsif (time_flag = '1' and in_time > stream_time) then
                time_error <= '1';
                if (after_wait_state = STATE_SAMPS_IN) then
                    state <= STATE_DRAIN_SAMPS;
                else
                    state <= STATE_STAT_OUT;
                end if;

            end if;

        when STATE_SAMPS_IN =>

            if (framed_fifo_out_valid = '1' and framed_fifo_out_ready = '1') then
                if (framed_fifo_out_last = '1') then
                    if (continuous_flag = '1') then
                        state <= STATE_HDR0_IN;
                    else
                        state <= STATE_STAT_OUT;
                        end_burst <= '1';
                    end if;
                elsif (underflow = '1') then
                    state <= STATE_DRAIN_SAMPS;
                end if;
            end if;

        when STATE_DRAIN_SAMPS =>
            --drain input until end of frame last signal
            if (framed_fifo_out_valid = '1' and framed_fifo_out_ready = '1') then
                if (framed_fifo_out_last = '1') then
                    state <= STATE_STAT_OUT;
                end if;
            end if;

        when STATE_STAT_OUT =>
            if (stat_fifo_in_valid = '1' and stat_fifo_in_ready = '1') then
                state <= STATE_TX_IDLE;
            end if;

        end case;
        end if;
    end if;

    end process;

    --my_icon: entity work.chipscope_icon
    --port map (
        --CONTROL0 => CONTROL_ILA
    --);
    --my_ila: entity work.chipscope_ila
    --port map (
        --CLK => clk,
        --CONTROL => CONTROL_ILA,
        --TRIG0 => TRIG_ILA,
        --DATA => DATA_ILA
    --);

    --TRIG_ILA(0) <= framed_fifo_out_last;
    --TRIG_ILA(1) <= framed_fifo_out_valid;
    --TRIG_ILA(2) <= framed_fifo_out_ready;
    ----TRIG_ILA(3) <= stat_tlast;
    --TRIG_ILA(4) <= stat_fifo_in_valid;
    --TRIG_ILA(5) <= stat_fifo_in_ready;
    --TRIG_ILA(6) <= dac_tready;
    --TRIG_ILA(7) <= '1' when dac_active_i else '0';

    --DATA_ILA(7 downto 0) <= std_logic_vector(state_num);
    --DATA_ILA(8) <= dac_fifo_out_valid;
    --DATA_ILA(9) <= dac_tready;
    --DATA_ILA(10) <= dac_fifo_in_valid;
    --DATA_ILA(11) <= dac_fifo_in_ready;
    --DATA_ILA(12) <= stat_fifo_in_valid;
    --DATA_ILA(13) <= stat_fifo_in_ready;
    --DATA_ILA(14) <= framed_fifo_out_last;
    --DATA_ILA(15) <= framed_fifo_out_valid;
    --DATA_ILA(16) <= framed_fifo_out_ready;
    --DATA_ILA(17) <= '1' when dac_active_i else '0';
    --DATA_ILA(18) <= time_flag;
    --DATA_ILA(19) <= continuous_flag;
    --DATA_ILA(20) <= underflow;
    --DATA_ILA(21) <= time_error;
    --DATA_ILA(22) <= end_burst;
    --DATA_ILA(23) <= '1' when time_wait0 else '0';

end architecture rtl;
