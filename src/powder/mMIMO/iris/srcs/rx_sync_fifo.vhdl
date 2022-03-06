------------------------------------------------------------------------
-- A fifo that aligns rx samples with rx events
-- This is a convenience block to nicely split
-- the various signals given a single data fifo.
------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity rx_sync_fifo is
    port(
        -- The clock domain
        clk : in std_logic;
        rst : in std_logic;

        -- input fifo signals
        in_samps : in std_logic_vector;
        in_trigger: in boolean;
        in_time : in unsigned;
        in_valid : in std_logic;
        in_ready : out std_logic;

        -- output fifo signals
        out_samps : out std_logic_vector;
        out_trigger : out boolean;
        out_time : out unsigned;
        out_valid : out std_logic;
        out_ready : in std_logic
    );
end entity rx_sync_fifo;

architecture rtl of rx_sync_fifo is

    signal in_data : std_logic_vector(in_time'length + in_samps'length downto 0);
    signal out_data : std_logic_vector(out_time'length + out_samps'length downto 0);

begin

    in_data(in_samps'range) <= in_samps;
    in_data(in_data'left-1 downto in_samps'length) <= std_logic_vector(in_time);
    in_data(in_data'left) <= '1' when in_trigger else '0';

    out_samps <= out_data(out_samps'range);
    out_time <= unsigned(out_data(out_data'left-1 downto out_samps'length));
    out_trigger <= out_data(out_data'left) = '1';

    fifo: entity work.StreamFifoSrl32
    port map (
        clk => clk,
        rst => rst,
        in_data => in_data,
        in_valid => in_valid,
        in_ready => in_ready,
        out_data => out_data,
        out_valid => out_valid,
        out_ready => out_ready
    );

end architecture rtl;
