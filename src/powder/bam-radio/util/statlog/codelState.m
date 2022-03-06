function [t,lavg,dropping,qs] = codelState(state,interval)

ctb = min([state.eventTime]);
ctbi = 1;
t = [];
lavg = [];
dropping = [];
qs = [];
for i=1:numel(state)
    ctc = state(i).eventTime;
    while (ctc-ctb) > interval
        ctbi = ctbi + 1;
        ctb = state(ctbi).eventTime;
    end
    t = [t ctc];
    lavg = [lavg state(i).avg_latency];
    dropping = [dropping state(i).dropping];
    qs = [qs state(i).queue_size];
end

end

