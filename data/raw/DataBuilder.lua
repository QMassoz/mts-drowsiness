require 'paths'
require 'image'
require 'xlua'
local DataBuilder = torch.class('DataBuilder')

-- NEEDS TO BE DEFINED
function DataBuilder:faceSize()
    return 100
end
function DataBuilder:nPoints()
    error('DataBuilder:nPoints() needs to be defined!')
    return 18
end
function DataBuilder:imageSize()
    error('DataBuilder:imageSize() needs to be defined!')
    return 640,480
end
function DataBuilder:listFilesForSubject(main_folder,subject)
    error('DataBuilder:listFilesForSubject() needs to be defined!')
    return {'img0','img1'}
end
function DataBuilder:loadLandmarks(main_folder,subject,filename)
    error('DataBuilder:loadLandmarks() needs to be define!')
    return torch.Tensor(self:nPoints()*2) -- format to [x1 x2 x3 ... y1 y2 y3 ...], 1-indexed
end
function DataBuilder:loadImage(main_folder,subject,filename)
    error('DataBuilder:loadImage() needs to be defined!')
    return torch.Tensor(128,128)
end
function DataBuilder:faceCenter(shapes2D)
    error('DataBuilder:faceCenter() needs to be defined!')
    local xc,yc,length = torch.Tensor(100,1), torch.Tensor(100,1), torch.Tensor(100,1)
    return xc,yc,length
end
function DataBuilder:eyeCenters(shapes2D)
    error('DataBuilder:eyeCenters() needs to be defined!')
    local xc,yc = torch.Tensor(100,2), torch.Tensor(100,2), torch.Tensor(100,2)
    return xc,yc,length
end
-- PREDEFINED
function DataBuilder:extractMultiPatches(img,shape2D,xc,yc,width,facePixels, owidth,oheight)
    local nCrops = xc:nElement()
    local oimages = torch.FloatTensor(nCrops,1,oheight,owidth)
    local oshapes2D = torch.FloatTensor(nCrops,2*self:nPoints())
    local extracted_boxes = {}
    for i=1,nCrops do
        oimages[i], oshapes2D[i], extracted_boxes[#extracted_boxes+1] = self:extractPatch(img,shape2D,xc[i],yc[i],width[i],facePixels, owidth,oheight)
    end
    return oimages,oshapes2D,extracted_boxes
end
local function clamp(x, min,max)
    if x >= max then
        return max
    elseif x <= min then
        return min
    else
        return x
    end
end
function DataBuilder:extractPatch(img,shape2D, xc,yc,width,facePixels, owidth, oheight)
    -- window
    local w = width/facePixels*owidth
    local h = width/facePixels*oheight
    local left = clamp(torch.round(xc-w/2), 1, img:size(2)-torch.round(w))
    local top = clamp(torch.round(yc-h/2), 1, img:size(1)-torch.round(h))
    w,h = torch.round(w),torch.round(h)
    -- extract window and scale it
    local img = image.scale(img[{{top,top+h},{left,left+w}}], owidth, oheight, 'bilinear')
    -- scale shape2D
    local N = self:nPoints()
    local zero_index = shape2D:eq(0)
    local shape2D = shape2D:clone()
    shape2D[{{1, N}}]:csub(left):mul(owidth / w):add(1)
    shape2D[{{N+1,2*N}}]:csub(top):mul(oheight / h):add(1)
    shape2D[zero_index] = 0
    --output
    return img, shape2D, {left,top,w,h}
end
function DataBuilder:detectVJ_face(img)
    if not self.face_detector then
        cv = require 'cv'
        require 'cv.objdetect'
        local haarfile = paths.concat(paths.dirname(paths.thisfile()), 'haarcascade.xml')
        self.face_detector = cv.CascadeClassifier{filename=haarfile}
        self.minSize = self:faceSize()/3
        self.maxSize = self:faceSize()*3
    end
    local h,w = img:size(1), img:size(2)
    local img = img:clone():mul(255):view(h,w,1):expand(h,w,3):byte()
    local faces, tensor = self.face_detector:detectMultiScale2{
        image   = img, 
        minSize = self.minSize,
        maxSize = self.maxSize
    }
    -- return biggest face
    local x,y,w,N = 0,0,0,tensor:nElement()
    if N==0 then
        x,y,w = 0,0,0
    elseif N==1 then
        x,y,w = faces.data[1].x, faces.data[1].y, faces.data[1].width
    else
        local x,y,w = -1,-1,-1
        for j=1,N do
            if faces.data[j].width > w then
                x,y,w = faces.data[j].x, faces.data[j].y, faces.data[j].width
            end
        end
    end
    if w > 0 then
        self.minSize = w/2
        self.maxSize = w*2
    else
        self.minSize = self.minSize/1.2
        self.maxSize = self.maxSize*1.2
    end
    return {x,y,w,w}
end


-- IN COMMON
function DataBuilder:__init(main_folder, datasetname, subjects, imageWidth, imageHeight, imageCenterType, dynamicExtraction, extractedFaceWidth)
	-- Initialization
	local main_folder = main_folder or paths.concat(paths.dirname(paths.thisfile()), datasetname)
	if not paths.dirp(main_folder) then
        error(main_folder .. ' does not exist!')
    end
    local imageWidth = imageWidth
    local imageHeight = imageHeight
    local imageCenterType = imageCenterType or 'face'
    local dynamicExtraction = dynamicExtraction or 'no'
    local extractedFaceWidth = extractedFaceWidth or 100

    print(sys.COLORS.green .. '>> building ' .. datasetname .. ' dataset <<' .. sys.COLORS.white)

    -- 1) load raw 2D shapes
    print('1) loading 2D shapes:')
    local shapes2D = {} -- all annotated shapes
    local nPoints = self:nPoints()
    local shapes_per_subject = {}   -- # of shapes for each valid subject
    local nSubjects = 0
    for i, s in ipairs(subjects) do
        xlua.progress(i,#subjects) -- progress bar
		local fnames = self:listFilesForSubject(main_folder, s)
		if next(fnames) ~= nil then -- fnames not empty
			shapes_per_subject[s] = #fnames
			nSubjects = nSubjects+1
			for _, f in pairs(fnames) do
                shapes2D[#shapes2D+1] = self:loadLandmarks(main_folder, s, f):totable()-- [x1 y1; x2 y2; ...]
			end
		end
    end
    shapes2D = torch.FloatTensor(shapes2D)
    local nExamples = shapes2D:size(1)

    -- shape/crop centers
    local xc, yc, width
    if imageCenterType == 'face' then
        xc, yc, width = self:faceCenter(shapes2D)
    elseif imageCenterType == 'eye' then -- [right eye, left eye] in the subject's POV (not the image's)
        xc, yc, width = self:eyeCenters(shapes2D)
    else
        error(imageCenterType .. ' is not a valid parameter!')
    end
    local nCrops = xc:size(2)
    assert(nCrops >= 1)

    -- 2) load images
 	print('2) loading images of ' .. nSubjects .. ' subjects:')
    local images = torch.FloatTensor(nExamples*nCrops, 1, imageHeight, imageWidth)
    local new_shapes2D = torch.FloatTensor(nExamples*nCrops,2*nPoints)
    local maxWidth, maxHeight = self:imageSize()
        -- variable for face detection
    local face_boxes = {}
    local extracted_face_boxes = {}

    --  box corners
    local tops = torch.round(yc-imageHeight/2):clamp(1,maxHeight-imageHeight+1)
    local lefts = torch.round(xc-imageWidth/2):clamp(1,maxWidth-imageWidth+1)
    local idx = 0;
    for _, s in ipairs(subjects) do
        local fnames = self:listFilesForSubject(main_folder, s)
        for _, f in pairs(fnames) do
           	collectgarbage()
            idx = idx + 1
            xlua.progress(idx,nExamples)
            local I = self:loadImage(main_folder,s,f)
            if imageCenterType == 'face' then
                table.insert(face_boxes, self:detectVJ_face(I))
            end
            if dynamicExtraction == 'no' then
                for c=1,nCrops do
                    images[{(idx-1)*nCrops+c,1}] = I:narrow(1,tops[{idx,c}],imageHeight):narrow(2,lefts[{idx,c}],imageWidth):clone()
        	    end

                if imageCenterType=='face' then
                    table.insert(extracted_face_boxes, {lefts[{idx,1}], tops[{idx,1}], imageWidth, imageHeight})
                end
            else
                local eb, indices = nil, {(idx-1)*nCrops+1,idx*nCrops}
                images[{indices,{},{},{}}], new_shapes2D[{indices,{}}], eb = self:extractMultiPatches(I,shapes2D[idx], xc[idx],yc[idx],width[idx], extractedFaceWidth, imageWidth,imageHeight)
                
                if imageCenterType=='face' then 
                    table.insert(extracted_face_boxes, eb[1])             
                end
            end
        end
    end

    lefts = lefts:view(-1)
    tops  = tops:view(-1)
    -- adapt shapes to cropped img
    if dynamicExtraction == 'no' then
        shapes2D = shapes2D:repeatTensor(1,nCrops):view(nCrops*shapes2D:size(1),shapes2D:size(2))
        
        local rep_lefts = lefts:view(-1,1):expand(nExamples*nCrops,nPoints)
        local rep_tops = tops:view(-1,1):expand(nExamples*nCrops,nPoints)
        local offset = torch.cat(rep_lefts, rep_tops, 2)
        local zero_index = shapes2D:eq(0)
        shapes2D:csub(offset):add(1)
        shapes2D[zero_index] = 0 -- values of 0 stay at 0
    else
        shapes2D = new_shapes2D
    end

    -- "sanity" check
    if idx ~= nExamples then
        error('Not all images were loaded !')
    end

    -- self fields
    self.inputs  = images:float()
    self.shapes2D = shapes2D:float()

    self.lefts = lefts:float()
    self.tops  = tops:float()
    self.shapeBox = {xc=xc,yc=yc,width=width}
    self.detected_box = torch.FloatTensor(face_boxes)
    self.extracted_box = torch.FloatTensor(extracted_face_boxes)
    self.imageCenterType = imageCenterType
    self.nExamples = nExamples
    self.nCrops = nCrops
    self.datasetname = datasetname
end
function DataBuilder:fuse(objects,keys)
    if keys then
        for _,k in ipairs(keys) do
            if torch.isTensor(self[k]) then
                local objects_k = {self[k]}
                for i=1,#objects do
                    table.insert(objects_k, objects[i][k])
                end
                self[k] = torch.cat(objects_k,1)
            else
                print('key ' .. k .. ' is not a tensor')
            end
        end
    elseif torch.type(objects) ~= 'table' then
        for k,v in pairs(self) do
            if torch.isTensor(v) then
                self[k] = torch.cat(v, objects[k], 1)
            end
        end
    end
end
function DataBuilder:scale(owidth, oheight)
    print(self.datasetname .. ': scaling to hxw = ' .. oheight .. 'x' .. owidth)
    local width, height = self.inputs:size(4), self.inputs:size(3)
    require 'image'
    -- scale inputs
    local new_inputs = torch.Tensor(self.inputs:size(1), self.inputs:size(2), oheight, owidth)
    for c=1,self.inputs:size(2) do
        new_inputs[{{},c,{},{}}] = image.scale(self.inputs[{{},c,{},{}}], owidth, oheight, 'bilinear')
    end
    self.inputs = new_inputs:typeAs(self.inputs)
    -- scale shapes2D
    local N = self:nPoints()
    local zero_index = self.shapes2D:eq(0)
    self.shapes2D[{{},{1, N}}]:csub(1):mul(owidth / width):add(1)
    self.shapes2D[{{},{N+1,2*N}}]:csub(1):mul(oheight / height):add(1)
    self.shapes2D[zero_index] = 0
    -- scale targets
    if self.targets then
        print('scaling targets, be careful!')
        self.targets:mul(owidth / width)
    end
end
function DataBuilder:remove_globalmean_inputs(mean)
    print(self.datasetname .. ': global normalization of the inputs')
    local mean = mean or {}
    local nChannels = self.inputs:size(2)
    -- compute mean
    if next(mean) == nil then
        for i=1,nChannels do
           mean[i] = self.inputs[{ {},i,{},{} }]:mean()
           self.inputs[{ {},i,{},{} }]:add(-mean[i])
        end
    -- use given mean
    else
        for i=1,nChannels do
            self.inputs[{ {},i,{},{} }]:add(-mean[i])
        end
    end
    return mean
end
function DataBuilder:remove_centralcrop_globalmean_inputs(mean, cropsize)
    print('<eye2eye-build> global normalization of the inputs')
    local mean = mean or {}
    local nChannels = self.inputs:size(2)
    -- compute mean
    if next(mean) == nil then
        for i=1,nChannels do
            local w = torch.ceil((self.inputs:size(4) - cropsize)/2)
            local h = torch.ceil((self.inputs:size(3) - cropsize)/2)
            mean[i] = self.inputs[{ {},i,{h+1,h+cropsize},{w+1,w+cropsize} }]:mean()
            self.inputs[{ {},i,{},{} }]:add(-mean[i])
        end
    -- use given mean
    else
        for i=1,nChannels do
            self.inputs[{ {},i,{},{} }]:add(-mean[i])
        end
    end
    return mean
end
function DataBuilder:normalize_inputs()
    print(self.datasetname .. ': individual normalization of the inputs')
    local x = self.inputs
    -- compute mean
    for k=1,x:size(1) do
    	for i=1,x:size(2) do
        	x[{ k,i,{},{} }]:csub(x[{ k,i,{},{} }]:mean())
        	x[{ k,i,{},{} }]:div(x[{ k,i,{},{} }]:std())
        end
    end
end
function DataBuilder:compute_face_detection_transform()
    local A = {}
    local B = {}
    for i=1,self.detected_box:size(1) do
        local x, y, w, h = self.extracted_box[{i,1}], self.extracted_box[{i,2}], self.extracted_box[{i,3}], self.extracted_box[{i,4}]
        local xi, yi, wi, hi = self.detected_box[{i,1}], self.detected_box[{i,2}], self.detected_box[{i,3}], self.detected_box[{i,4}]
        if not (xi<x or yi<y or xi+wi>x+w or yi+hi>y+h) and wi>0 then
            -- add example to A
            table.insert(A, {wi,0,0,0}) -- hypothesis that wi=hi
            table.insert(A, {0,wi,0,0})
            table.insert(A, {0,0,wi,0})
            table.insert(A, {0,0,0,wi})
            -- add example to B
            table.insert(B,{x-xi})
            table.insert(B,{y-yi})
            table.insert(B,{w})
            table.insert(B,{h})
        end
    end
    A=torch.Tensor(A)
    B=torch.Tensor(B)
    return torch.gels(B,A)
end

function DataBuilder:view(idx)
    if idx > self.inputs:size(1) then
        print(idx .. ' is greater than the number of examples')
        return
    end
	local cv = require 'cv'
	require 'cv.imgcodecs'
	require 'cv.imgproc'
	require 'cv.highgui'
	cv.namedWindow{'win_face'}

	local img = self.inputs[{idx,1,{},{}}]
	local h,w = img:size(1), img:size(2)
	local img = img:clone():mul(255):view(h,w,1):expand(h,w,3):byte()
	local landmarks = self.shapes2D[{idx,{}}]-1
	
	for i=1,self:nPoints() do
		cv.circle{img,center={landmarks[i],landmarks[i+self:nPoints()]}, radius=2, color={0,255,0},thickness=-1}
	end

	cv.imshow{'win_face', img}
	key = cv.waitKey{0}
	cv.destroyAllWindows()
end