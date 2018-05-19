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
require 'stn'

--require 'InstanceNormalization'
URnet = require 'URnet_rgb'

GPU_ID = 1;


local lowHd5 = hdf5.open('/media/anu-user1/4TB/Fatima/DestyleAttrTransfer/dataset/SF_Test_8k.hdf5', 'r')
--local lowHd5 = hdf5.open('/media/anu-user1/4TB/Fatima/DestyleAttrTransfer/dataset/SF_Test_STN_9k.hdf5', 'r')

local data_SF = lowHd5:read('YTC'):all()
data_SF:mul(2):add(-1)
lowHd5:close()
print(data_SF:size())
nval = data_SF:size(1)
num = nval
--trainData_SF = data_SF[{{1, ntrain}}]
valData_SF = data_SF[{{1,nval}}]

inter_dataset = torch.FloatTensor(num,3,128,128):fill(0)

cutorch.setDevice(GPU_ID + 1)
print('<gpu> using device ' .. GPU_ID)
torch.setdefaulttensortype('torch.CudaTensor')

function saveImages(data, model_STN, foldername, start)
  local N = data:size(1)
  local inputs_SF = torch.Tensor(N,3,128,128)
  
  for i = 1,N do
	inputs_SF[i]  = data[i]
  end
 
  -- Generate 
  --sys.tic()
  local samples = model_RF:forward(inputs_SF)
  local samples_RF = nn.HardTanh():forward(samples)
  
  local to_plot = {}
  for i = 1,N do
    to_plot[i] = samples_RF[i]:float()
    torch.setdefaulttensortype('torch.FloatTensor')
    local GEN = image.toDisplayTensor({input=to_plot[i], nrow=1})
    --GEN:add(1):div(2):float()
    GEN = GEN:index(1,torch.LongTensor{3,2,1})
	print(i)
    
    filename = string.format("%04d.png",i+start)
    image.save(foldername .. filename, GEN)
  end  

  torch.setdefaulttensortype('torch.CudaTensor') 
  cutorch.setDevice(GPU_ID + 1)  
  
  samples_RF = nn.HardTanh():forward(samples_RF)  
  samples_RF:add(1):mul(0.5)    
  for i = 1,N do
	local tmp = samples_RF[i]:float()
	inter_dataset[start+i] = tmp:clone()
  end   
   
end

torch.setdefaulttensortype('torch.CudaTensor')

--model = torch.load('/media/anu-user1/4TB/Fatima/GAN/old/destyle_End2End_4Styles_noDescriminative/model.net') -- for undiscriminative
--model_RF = model[1] -- for No discriminative 
--model = torch.load('/media/anu-user1/4TB/Fatima/GAN/destyle_percep_Unalign_STN_eta/adversarial_net_245') --for WACV
--model = torch.load('/media/anu-user1/4TB/Fatima/GAN/old/destyle_Dicta/adversarial_net_500') --for Dicta
--model = torch.load('/media/anu-user1/4TB/Fatima/GAN/destyle_percep_align_SK/adversarial_net_218') --- for Sketch
model = torch.load('/media/anu-user1/4TB/Fatima/GAN/old/Unet/destyle_4Style_Unet_old/adversarial_net_30') -- for pix2pix
model_RF = model.G  -- for discriminative
model_RF:evaluate()  -- for discriminative

foldername = '/media/anu-user1/4TB/Fatima/DestyleAttrTransfer/dataset/SF_Test_8k_al1/'
if not paths.dirp(foldername) then
	paths.mkdir(foldername)
end


-- save all the training dataset
num_remainder = num%100
num_loop = (num-num_remainder)/100
for i = 1,num_loop do
	saveImages(valData_SF[{{(i-1)*100+1,i*100}}], model_RF, foldername, (i-1)*100)
end
if num_remainder ~= 0 then
	saveImages(valData_SF[{{num_loop*100+1,num}}], model_RF, foldername, num_loop*100)
end

