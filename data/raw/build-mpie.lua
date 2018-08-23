--require '../DataBuilder'

require 'paths'
require 'xlua'
require 'image'
local cv = require 'cv'
require 'cv.objdetect'
require 'cv.imgcodecs'

local Builder,parent = torch.class('MPIEBuilder','DataBuilder')

function Builder:__init(subjects, imageWidth, imageHeight, imageCenterType, dynamicExtraction, extractedFaceWidth)
    parent.__init(self,paths.concat(paths.dirname(paths.thisfile()), 'mpie'), 'MPIE', subjects, imageWidth,imageHeight,imageCenterType, dynamicExtraction, extractedFaceWidth)
end
function Builder:faceSize()
    return 90*2.4
end
function Builder:imageSize()
	local width = 640
	local height = 480
	return width, height
end
function Builder:nPoints()
    return 68
end

local function array_concat(v1, v2)
-- concatenate 2 tables into 1
    local l = #v1
    for i,v in ipairs(v2) do
        v1[l+i] = v
    end
    return v1
end
function Builder:listFilesForSubject(main_folder,subject)
    require 'paths'
    local function valid(f)
        local s,e = f:find(string.format('%03d',subject))
        if s and e and s == 1 and e == 3 then
            return true
        else
            return false
        end
    end
    local filenames = {}
    for f in paths.files(main_folder .. '/points',valid) do
        filenames = array_concat(filenames, {paths.basename(f,'.pts')})
    end
    return filenames
end
function Builder:loadLandmarks(main_folder,subject, filename)
    require 'csvigo'
    local filename = paths.concat(main_folder, 'points', filename .. '.pts')
    local data = csvigo.load{path=filename,mode='raw', header=false, verbose=false, separator=' '}
    local points = {}
    for i = 4,71 do
        points[i-3] = data[i]
    end
    -- format to [x1 x2 x3 ... y1 y2 y3 ...]
    points = torch.Tensor(points):t():contiguous():view(-1)
    return points:add(1)
end
function Builder:loadImage(main_folder,subject,filename)
    local I = image.load(paths.concat(main_folder,'images', filename..'.png'),3,'float'):mean(1):squeeze()
    return I
end

local function delta(x)
    return x:max(2)-x:min(2)
end
function Builder:faceCenter(shapes)
    local xc = shapes:narrow(2,1,68):mean(2)
    local yc = shapes:narrow(2,69,68):mean(2)

    local width = delta(shapes:narrow(2,1,68))
    local height = delta(shapes:narrow(2,69,68))
    local length,_ = torch.cat(height,width,2):max(2)

    return xc, yc, length
end
function Builder:eyeCenters(shapes)
    local xc = torch.cat(shapes:index(2,torch.LongTensor{37,38,39,40,41,42}):mean(2),
            shapes:index(2,torch.LongTensor{43,44,45,46,47,48}):mean(2),2)
    local yc = torch.cat(shapes:index(2,torch.LongTensor{37,38,39,40,41,42}+68):mean(2),
            shapes:index(2,torch.LongTensor{43,44,45,46,47,48}+68):mean(2),2)

    local width = delta(shapes:narrow(2,1,68))
    local height = delta(shapes:narrow(2,69,68))
    local length,_ = torch.cat(height,width,2):max(2):repeatTensor(1,2)

    return xc, yc, length
end


local function get_distance(x1, y1, x2, y2)
    return ((x1-x2):pow(2) + (y1-y2):pow(2)):sqrt()
end
local function shape2eld(s,eye)
    if eye=='r' or eye=='right' then
        local d1 = get_distance(s:select(2,38),s:select(2,68+38),s:select(2,42),s:select(2,68+42))
        local d2 = get_distance(s:select(2,39),s:select(2,68+39),s:select(2,41),s:select(2,68+41))
        return (d1+d2)/2
    elseif eye=='l' or eye=='left' then
        local d1 = get_distance(s:select(2,44),s:select(2,68+44),s:select(2,48),s:select(2,68+48))
        local d2 = get_distance(s:select(2,45),s:select(2,68+45),s:select(2,47),s:select(2,68+47))
        return (d1+d2)/2
    else
        error('unknown eye input')
    end

end
function Builder:eyelids_distances()
    local r_eld = shape2eld(self.shapes2D,'right')
    local l_eld = shape2eld(self.shapes2D,'left')
	if self.imageCenterType == 'face' then
        return torch.cat(r_eld,l_eld,2)
	elseif self.imageCenterType == 'eye' then
        r_eld = r_eld:view(-1,2):select(2,1)
        l_eld = l_eld:view(-1,2):select(2,2)
        return torch.cat(r_eld,l_eld,2):view(-1,1)
	else
		error(self.imageCenterType .. ' is not a valid parameter!')
	end
end
function Builder:view(idx)
	local cv = require 'cv'
	require 'cv.imgcodecs'
	require 'cv.imgproc'
	require 'cv.highgui'
	cv.namedWindow{'win_face'}

	local img = self.inputs[{idx,1,{},{}}]
	local h,w = img:size(1), img:size(2)
	local img = img:clone():mul(255):view(h,w,1):expand(h,w,3):byte()
	local landmarks = self.shapes2D[{idx,{}}]
	
	for i=1,68 do
		cv.circle{img,center={landmarks[i],landmarks[i+68]}, radius=2, color={0,255,0},thickness=-1}
	end
    cv.line{img,pt1={landmarks[37],landmarks[37+68]},pt2={landmarks[38],landmarks[38+68]},color={0,255,0}}
    cv.line{img,pt1={landmarks[38],landmarks[38+68]},pt2={landmarks[39],landmarks[39+68]},color={0,255,0}}
    cv.line{img,pt1={landmarks[39],landmarks[39+68]},pt2={landmarks[40],landmarks[40+68]},color={0,255,0}}
    cv.line{img,pt1={landmarks[40],landmarks[40+68]},pt2={landmarks[41],landmarks[41+68]},color={0,255,0}}
    cv.line{img,pt1={landmarks[41],landmarks[41+68]},pt2={landmarks[42],landmarks[42+68]},color={0,255,0}}
    cv.line{img,pt1={landmarks[42],landmarks[42+68]},pt2={landmarks[37],landmarks[37+68]},color={0,255,0}}

    cv.line{img,pt1={landmarks[43],landmarks[43+68]},pt2={landmarks[44],landmarks[44+68]},color={0,255,0}}
    cv.line{img,pt1={landmarks[44],landmarks[44+68]},pt2={landmarks[45],landmarks[45+68]},color={0,255,0}}
    cv.line{img,pt1={landmarks[45],landmarks[45+68]},pt2={landmarks[46],landmarks[46+68]},color={0,255,0}}
    cv.line{img,pt1={landmarks[46],landmarks[46+68]},pt2={landmarks[47],landmarks[47+68]},color={0,255,0}}
    cv.line{img,pt1={landmarks[47],landmarks[47+68]},pt2={landmarks[48],landmarks[48+68]},color={0,255,0}}
    cv.line{img,pt1={landmarks[48],landmarks[48+68]},pt2={landmarks[43],landmarks[43+68]},color={0,255,0}}

	cv.imshow{'win_face', img}
	key = cv.waitKey{0}
	cv.destroyAllWindows()
end

function Builder:test()
    LS=PUTBuilder(torch.range(1,100):totable(),256,256,'eye') 
    LS:scale(64,64)
    LS.targets = LS:eyelids_distances()
    LS:normalize_inputs()
    LS:view(1)
end
