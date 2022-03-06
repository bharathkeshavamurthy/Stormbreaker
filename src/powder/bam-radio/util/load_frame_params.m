% load the protobuffed version of the OFDMFrameparams struct (see
% src/debug.proto) as a MATLAB struct
%
% Author: Dennis Ogbe <dogbe@purdue.edu>
%
% Use at own risk

function [out, raw] = load_frame_params(filename)

warning('off', 'MATLAB:structonObject');

% load the python protobuf object
fp = py.debug_pb2.DFTSOFDMFrameParams();
f = py.open(filename);
b = f.read();
fp.ParseFromString(b);
f.close();
raw = fp;

% convert the object to a MATLAB struct
out = struct();
fields = {'num_symbols', 'num_tx_samples', 'num_bits', 'dft_spread_length'};
for ii=1:length(fields)
    eval(['out.' fields{ii} '= double(fp.' fields{ii} ');';]);
end

% this is dirty but works
stmp1 = cell(fp.symbols);
for ii=1:length(stmp1)
    stmp2(ii) = struct(stmp1{ii});
end
% scalar fields
sfields = {
    'symbol_length',
    'oversample_rate',
    'cyclic_prefix_length',
    'postfix_pad',
    'num_tx_samples',
    'num_data_carriers',
    'num_bits',
    'num_pilot_carriers'
};
% carrier mappings (simple vector fields with 1+ re-indexing)
svfields = {'data_carrier_mapping', 'pilot_carrier_mapping'};
% complex number fields
cfields = {'pilot_symbols', 'prefix'};
% go through fields of struct and manually convert
for ii=1:length(stmp2)
    for jj=1:length(sfields)
        eval(['out.symbols(' num2str(ii) ').' sfields{jj} ' = double(stmp2(' num2str(ii) ').' sfields{jj} ');' ]);
    end
    for jj=1:length(svfields)
        tmp1 = cell(getfield(stmp2(ii), svfields{jj}));
        % n.b. re-index for 1-based
        eval(['out.symbols(' num2str(ii) ').' svfields{jj} ' = cellfun(@double, tmp1) + 1;']);
    end
    for jj=1:length(cfields)
        tmp1 = cell(getfield(stmp2(ii), cfields{jj}));
        tmp2 = 1+1j * ones(size(tmp1));
        for kk=1:length(tmp2)
            tmp3 = struct(tmp1{kk});
            tmp2(kk) = tmp3.re + 1j * tmp3.im;
        end
        eval(['out.symbols(' num2str(ii) ').' cfields{jj} ' = tmp2;']);
    end
end

warning('on', 'MATLAB:structonObject');