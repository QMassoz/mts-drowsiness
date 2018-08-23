require 'torch'
require 'csvigo'
require 'paths'
require 'image'
require 'xlua'


local M = {}
function M.build(opt, cacheFile)
    torch.manualSeed(5)
    torch.setnumthreads(1)
    
    print("<ieyes2eld> building data...")
    
    -- init
    LS,VS,TS = {},{},{}
    local inputSize = {1,24,24} -- {#channels, width, height}
    local kinect_randperm = torch.randperm(16)
    local mpie_randperm = torch.randperm(346)
    local put_randperm = torch.randperm(100)

    local dynamicExtraction = 'yes'
    local facePixelWidth = 70

    require 'data/raw/DataBuilder'

    -- kinect data
    require 'data/raw/build-kinect'
    local TS_kinect = ULGBuilder(kinect_randperm[{{1,3}}]:totable(), inputSize[2],inputSize[3], 'eye', dynamicExtraction, facePixelWidth)
    local VS_kinect = ULGBuilder(kinect_randperm[{{4,5}}]:totable(), inputSize[2],inputSize[3], 'eye', dynamicExtraction, facePixelWidth)
    local LS_kinect = ULGBuilder(kinect_randperm[{{6,-1}}]:totable(), inputSize[2],inputSize[3], 'eye', dynamicExtraction, facePixelWidth)
    LS_kinect.targets = LS_kinect:eyelids_distances()
    VS_kinect.targets = VS_kinect:eyelids_distances()
    TS_kinect.targets = TS_kinect:eyelids_distances()

    -- mpie data
    require 'data/raw/build-mpie'
    local mpie_scale = dynamicExtraction=='yes' and 1 or 2.4
    local TS_mpie = MPIEBuilder(mpie_randperm[{{1,67}}]:totable(), mpie_scale*inputSize[2], mpie_scale*inputSize[3],'eye', dynamicExtraction, facePixelWidth)
    local VS_mpie = MPIEBuilder(mpie_randperm[{{68,95}}]:totable(), mpie_scale*inputSize[2], mpie_scale*inputSize[3],'eye', dynamicExtraction, facePixelWidth)
    local LS_mpie = MPIEBuilder(mpie_randperm[{{96,-1}}]:totable(), mpie_scale*inputSize[2], mpie_scale*inputSize[3],'eye', dynamicExtraction, facePixelWidth)
    if dynamicExtraction == 'no' then
        LS_mpie:scale(inputSize[2],inputSize[3])
        VS_mpie:scale(inputSize[2],inputSize[3])
        TS_mpie:scale(inputSize[2],inputSize[3])
    end
    LS_mpie.targets = LS_mpie:eyelids_distances()
    VS_mpie.targets = VS_mpie:eyelids_distances()
    TS_mpie.targets = TS_mpie:eyelids_distances()

    -- LS, VS, and TS
    LS_kinect:fuse({LS_mpie,LS_put}, {'inputs', 'targets'})
    VS_kinect:fuse({VS_mpie,VS_put}, {'inputs', 'targets'})

    -- Normalize inputs
    local mean = LS_kinect:remove_globalmean_inputs()
    VS_kinect:remove_globalmean_inputs(mean)
    TS_kinect:remove_globalmean_inputs(mean)

    -- format
    LS = {inputs=LS_kinect.inputs, targets=LS_kinect.targets, mean=mean}
    VS = {inputs=VS_kinect.inputs, targets=VS_kinect.targets}
    TS = {inputs=TS_kinect.inputs, targets=TS_kinect.targets}

    
    local cacheFile = cacheFile or 'data/processed/ieyes2eld.t7'
    print('<ieyes2eld> saving binary ' .. cacheFile)
    torch.save(cacheFile, {
        train=LS,
        valid=VS,
        test=TS,
        inputSize=inputSize
    })
end

return M