------------------------------------------------------------
--for adding perception loss you should add require 'PerceptionLoss'
--require 'preprocess' and change URnet and add vgg_model = createVggmodel() and
--PerceptionLoss = nn.PerceptionLoss(vgg_model, 4):cuda(). in preprocess you should choose layers for perception loss!
------------------------------------------------------------

require 'hdf5'
require 'nngraph'
require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'cudnn'
require 'PerceptionLoss'
require 'preprocess'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
URnet = require 'adverserial_xin_v1_D_percep'

--URnet = require 'UR_rgb'
stn_L1 = require 'stn_L1'
stn_L2 = require 'stn_L2'
stn_L3 = require 'stn_L3'
--stn_L4 = require 'stn_L4'
--stn_L5 = require 'stn_L5'
--stn_L6 = require 'stn_L6'

----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "destyle_percep_Unalign_STN")      subdirectory to save logs
  --saveFreq         (default 1)           save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.001)       learning rate
  -b,--batchSize     (default 32)          batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  -d,--noiseDim      (default 512)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 128)         scale of images to train on
  --lambda           (default 0.01)        trade off D and Euclidean distance 
  --margin           (default 0.3)         trade off D and G   
  --eta          	 (default 0.01)       trade off perception and Euclidean distance  
]]


if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

print(opt)

ntrain = 29952
nval   = 384

local highHd5 = hdf5.open('datasets/Real3_Align.hdf5', 'r')
local data_Real = highHd5:read('YTC'):all()
data_Real:mul(2):add(-1)
highHd5:close()
trainData_Real = data_Real[{{1, ntrain}}]
valData_Real = data_Real[{{ntrain+1, nval+ntrain}}]


local lowHd5 = hdf5.open('datasets/3styles_Un.hdf5', 'r')
local data_SF = lowHd5:read('YTC'):all()
data_SF:mul(2):add(-1)
lowHd5:close()
trainData_SF = data_SF[{{1, ntrain}}]
valData_SF = data_SF[{{ntrain+1, nval+ntrain}}]

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
	cutorch.setDevice(opt.gpu + 1)
	print('<gpu> using device ' .. opt.gpu)
	torch.setdefaulttensortype('torch.CudaTensor')
else
	torch.setdefaulttensortype('torch.FloatTensor')
end

input_scale  = 128
opt.scale = valData_Real:size(4)
--print(opt.scale)
opt.geometry = {3, opt.scale, opt.scale}

if opt.network == '' then
	model_D = nn.Sequential()
	model_D:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
	model_D:add(cudnn.SpatialMaxPooling(2,2)) --64x64
	model_D:add(cudnn.SpatialBatchNormalization(32))
	model_D:add(nn.LeakyReLU(0.2, true))
	--model_D:add(nn.SpatialDropout(0.2))  
	model_D:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
	model_D:add(cudnn.SpatialMaxPooling(2,2)) --32x32
	model_D:add(cudnn.SpatialBatchNormalization(64))
	model_D:add(nn.LeakyReLU(0.2, true))
	--model_D:add(nn.SpatialDropout(0.2))
	model_D:add(cudnn.SpatialConvolution(64, 128, 5, 5, 1, 1, 2, 2))
	model_D:add(cudnn.SpatialMaxPooling(2,2))  --16x16
	model_D:add(cudnn.SpatialBatchNormalization(128))
	model_D:add(nn.LeakyReLU(0.2, true))
	--model_D:add(nn.SpatialDropout(0.2))
	model_D:add(cudnn.SpatialConvolution(128, 96, 3, 3, 1, 1, 1, 1))
	model_D:add(cudnn.SpatialMaxPooling(2,2))  --8x8
	model_D:add(cudnn.SpatialBatchNormalization(96))
	model_D:add(nn.LeakyReLU(0.2, true))
	--model_D:add(nn.SpatialDropout(0.2))
	model_D:add(nn.Reshape(8*8*96))
	model_D:add(nn.Linear(8*8*96, 1024))
	model_D:add(nn.LeakyReLU(0.2, true))
	--model_D:add(nn.Dropout())  -- in this case, only dropout is removed here.
	model_D:add(nn.Linear(1024,1))
	model_D:add(nn.Sigmoid())
	
	----------------------------------------------------------------------
    model_G = nn.Sequential()
	model_G:add(cudnn.SpatialConvolution(3, 16, 5, 5, 1, 1, 2, 2))
	model_G:add(cudnn.SpatialBatchNormalization(16))
	model_G:add(nn.LeakyReLU(0.2, true))    
	model_G:add(cudnn.SpatialMaxPooling(2,2))  --64*64

	model_G:add(cudnn.SpatialConvolution(16, 32, 5, 5, 1, 1, 2, 2))
	model_G:add(cudnn.SpatialBatchNormalization(32))
	model_G:add(nn.LeakyReLU(0.2, true)) 
	model_G:add(stn_L1)   
	model_G:add(cudnn.SpatialMaxPooling(2,2))  -- 32*32
	   
	model_G:add(cudnn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(64))
	model_G:add(nn.LeakyReLU(0.2, true))
	model_G:add(stn_L2)
	model_G:add(cudnn.SpatialMaxPooling(2,2)) --16*16
	
	model_G:add(cudnn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(128))
	model_G:add(nn.LeakyReLU(0.2, true))
	model_G:add(cudnn.SpatialMaxPooling(2,2)) --8*8
	
	model_G:add(cudnn.SpatialConvolution(128, 256, 1, 1, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(256))
	model_G:add(nn.LeakyReLU(0.2, true))
	model_G:add(cudnn.SpatialMaxPooling(2,2)) --4*4
	
	model_G:add(cudnn.SpatialConvolution(256, 512, 1, 1, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(512))
	model_G:add(nn.LeakyReLU(0.2, true))
	model_G:add(cudnn.SpatialMaxPooling(2,2)) --2*2
	
	model_G:add(nn.Reshape(2*2*512))
	model_G:add(nn.Linear(2*2*512,2*2*512))
	model_G:add(nn.View(512,2,2))
	
	model_G:add(cudnn.SpatialConvolution(512, 256, 1, 1, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(256))
	model_G:add(cudnn.ReLU(true))
	model_G:add(nn.SpatialUpSamplingNearest(2)) -- 4*4
	
	model_G:add(cudnn.SpatialConvolution(256, 128, 3, 3, 1, 1, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(128))
	model_G:add(cudnn.ReLU(true))
	model_G:add(nn.SpatialUpSamplingNearest(2)) -- 8*8
	model_G:add(stn_L3)
	
	model_G:add(cudnn.SpatialConvolution(128, 64, 3, 3, 1, 1, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(64))
	model_G:add(cudnn.ReLU(true))
	model_G:add(nn.SpatialUpSamplingNearest(2)) -- 16*16
--	model_G:add(stn_L4)
	
	model_G:add(cudnn.SpatialConvolution(64, 512, 3, 3, 1, 1, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(512))
	model_G:add(cudnn.ReLU(true))  
	model_G:add(nn.SpatialUpSamplingNearest(2))  --32*32
--	model_G:add(stn_L5)  
	model_G:add(cudnn.SpatialConvolution(512,256, 3, 3, 1, 1, 1, 1))
	model_G:add(cudnn.SpatialBatchNormalization(256))
	model_G:add(cudnn.ReLU(true))  
	model_G:add(nn.SpatialUpSamplingNearest(2))   --64*64
--	model_G:add(stn_L6)    
	model_G:add(cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2))
	model_G:add(cudnn.SpatialBatchNormalization(128))
	model_G:add(cudnn.ReLU(true))
	model_G:add(nn.SpatialUpSamplingNearest(2))

	model_G:add(cudnn.SpatialConvolution(128, 64, 5, 5, 1, 1, 2, 2))  
	model_G:add(cudnn.SpatialBatchNormalization(64))
	model_G:add(cudnn.ReLU(true))
	model_G:add(cudnn.SpatialConvolution(64, 3, 5, 5, 1, 1, 2, 2))
else
	print('<trainer> reloading previously trained network: ' .. opt.network)
	tmp = torch.load(opt.network)
	model_D = tmp.D  
	model_G = tmp.G
	epoch = 110  
end

print('Copy model to gpu')
model_D:cuda()
model_G:cuda()  -- convert model to CUDA

-- loss function: negative log-likelihood
criterion_D = nn.BCECriterion()
criterion_G = nn.MSECriterion()

vgg_model = createVggmodel()
PerceptionLoss = nn.PerceptionLoss(vgg_model, 4):cuda()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Training parameters
sgdState_D = {
	learningRate = opt.learningRate,
	momentum = opt.momentum,
	optimize=true,
	numUpdates = 0
}
sgdState_G = {
	learningRate = opt.learningRate,
	momentum = opt.momentum,
	optimize=true,
	numUpdates=0
}
-- Get examples to plot
function getSamples(dataset, N, M, K)
	local N = N or 10
	local M = M 
	local K = K 
	local dataset_Real = dataset
	local inputs   = torch.Tensor(N+M,3,128,128)
	for i = 1,N do 
		--idx = math.random(nval)
		--inputs[i] = image.scale(torch.squeeze(dataset_Real[i]),16,16)
		inputs[i] = dataset[i]
	end
	for i = 1,M do 
		--idx = math.random(nval)
		--inputs[i] = image.scale(torch.squeeze(dataset_Real[i]),16,16)
		inputs[i+N] = dataset[i+K]
	end
	
	local samples = model_G:forward(inputs)
	samples = nn.HardTanh():forward(samples)
	local to_plot = {}
	for i = 1,(N+M) do 
		to_plot[#to_plot+1] = samples[i]:float()
	end
	return to_plot
end

while true do 
	local to_plot = getSamples(valData_SF,50,50,128)
	
	torch.setdefaulttensortype('torch.FloatTensor')
	
	trainLogger:style{['MSE accuarcy1'] = '-'}
	-- trainLogger:style{['MSE accuarcy'] = '-'}
	trainLogger:plot()
	
	local formatted = image.toDisplayTensor({input = to_plot, nrow = 10})
	formatted:float()
	formatted = formatted:index(1,torch.LongTensor{3,2,1})
	
	image.save(opt.save .. '/UR_example_' .. (epoch or 0) .. '.png', formatted)
	
	IDX = torch.randperm(ntrain)
	
	if opt.gpu then 
		torch.setdefaulttensortype('torch.CudaTensor')
	else
		torch.setdefaulttensortype('torch.FloatTensor')
	end
	
	URnet.train(trainData_SF,trainData_Real)
	
	sgdState_D.momentum = math.min(sgdState_D.momentum + 0.0008, 0.7)
    sgdState_D.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
	
	sgdState_G.momentum = math.min(sgdState_G.momentum + 0.0008, 0.7)
	sgdState_G.learningRate = math.max(opt.learningRate*0.99^epoch, 0.000001)
	
	opt.lambda = math.max(opt.lambda*0.995, 0.005)   -- or 0.995
--	opt.eta = math.max(opt.eta*0.995, 0.0005)
--	opt.eta = math.max(opt.eta*1.005, 0.0005)
end