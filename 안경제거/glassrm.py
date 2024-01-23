"""
Architecture of Shadow/Glass Mask Network and De-Shadow/Glass Network
Note: The code is modified based on
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from model.resnet import ResnetBlock

def make_model(args, parent=False):
    return GlassRMNet(args)

def pad_image(image):
    original_h, original_w = image.size(1), image.size(2)
    target_h = original_h + 128 - original_h%128 
    target_w = original_w + 128 - original_w%128 
        
    pad_h = target_h - original_h
    pad_w = target_w - original_w
            
    padded_image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h))
    return padded_image

def unpad_image(padded_image, original_h, original_w):
    unpadded_image = padded_image[:, :, :original_h, :original_w]
    return unpadded_image


def out2mask(tensor, soft=True, scale=1.25):
    if soft:
        ret = (nn.Softmax2d()(tensor)[:, 1:2, :, :]) * scale
    else:
        ret = (tensor.argmax(1).unsqueeze(1).float())
    return ret


class GlassRMNet(nn.Module):

    def __init__(self, args):
        super(GlassRMNet, self).__init__()

        self.model_gMask = ResnetGeneratorMask(3, 2)
        self.model_deshadow = ResnetGenerator(4, 3)
        self.model_deglass = ResnetGenerator(4, 3)
        self.model_flowMap = ResnetGenerator(4, 2)
        #도수에 맞게 확대하는 모델
        # self.model_lensaware = ResnetGenerator(args)
        # self.model_lensmag = ResnetGenerator(args)

    #try5 여기서 오류 남
    #def forward(self, input):
    def forward(self, input):
        # flowMap = input[1]
        # input = input[0]

        if not self.training:
            _,_,h,w = input.shape
            size_limit = 512
            if h > size_limit or w > size_limit:
                input =  torch.nn.functional.interpolate(input, size=[size_limit,size_limit], mode='bilinear')
        # original_h, original_w = input.size(2), input.size(3)
        # input = pad_image(input)

        output_gMask = self.model_gMask(input)
        # output_lensaware = self.model_lensaware(input)
        output_deshadow = self.model_deshadow(torch.cat([input, output_gMask],dim=1))
        output_deshadow_masked = output_deshadow * (1-output_gMask)
        # output_deglass = self.model_deglass(torch.cat([output_deshadow_masked, output_gMask],dim=1))

        #try3
        #복원할 때 flowmap 사용
        ###############################
        output_flowMap = self.model_flowMap(torch.cat([input, output_gMask],dim=1))

        ##T
        # print(flowMap.shape)
        # output_flowMap = flowMap[:,0:2,:,:]

        output_flowMap_grid = (output_flowMap*255+127)

        output_flowMap_grid = output_flowMap_grid.to('cuda')  #추가코드
        

        #try5 : input 텐서와 output_flowMap_grid의 차원이 통일되지 않는다는 내용의 에러
        #input_expanded = input.unsqueeze(1)

        output_flowMap_grid[:,0:1,:,:] = output_flowMap_grid[:,0:1,:,:]/input.shape[2]
        output_flowMap_grid[:,1:2,:,:] = output_flowMap_grid[:,1:2,:,:]/input.shape[3]


        #2, 3 바꾸기
        x_space = torch.linspace(-1,1,input.shape[2])
        y_space = torch.linspace(-1,1,input.shape[3])
        meshx, meshy = torch.meshgrid((x_space, y_space))
        flow_map = torch.stack((meshy, meshx), 2)


        flow_map = flow_map.to('cuda') #추가코드
        
        flow_map = flow_map - output_flowMap_grid[:, 0:2, :, :].permute(0, 2, 3, 1) # note: check flow_map.shape 
        
        #flow_map = flow_map.permute(1, 2, 0) - output_flowMap_grid[:, 0:2, :, :] #0912수정

        
        output_deglass = F.grid_sample(output_deshadow_masked, flow_map.float(), mode='bilinear', align_corners=True)
        ############################

        output_deglass = self.model_deglass(torch.cat([output_deglass, output_gMask],dim=1))
        
        #try2
        #output_flowMap = self.model_flowMap

        ######
        output_deglass = self.model_deglass(torch.cat([output_deshadow_masked, output_gMask],dim=1))

        # output_final = self.model_lensmag(self.model_gMask, self.model_deglassMask)
        output_final = output_deglass

        if not self.training:
            if h > size_limit or w > size_limit:
                output_final =  torch.nn.functional.interpolate(output_final, size=[h,w], mode='bilinear')
        # output = unpad_image(output, original_h, original_w)0
        return output_final, output_gMask, output_deshadow , output_flowMap #,  None, None

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    
    def __init__(self, input_nc, output_nc):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        ngf=64
        norm_layer=nn.BatchNorm2d
        use_dropout=False
        n_blocks=6
        padding_type='reflect'

        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        # input = (input - 128)/128
        # output = self.model(input)*128 + 128
        # original_h, original_w = input.size(2), input.size(3)
        # input = pad_image(input)
        output = self.model(input) # *255
        # output = unpad_image(output, original_h, original_w)
        return output


class ResnetGeneratorMask(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGeneratorMask, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        self.model = nn.Sequential(*model)
        

    def forward(self, input):
        """Standard forward"""
        # original_h, original_w = input.size(2), input.size(3)
        # padded_image = pad_image(input)
        # output = self.model(padded_image)
        output = self.model(input)
        # output = unpad_image(output, original_h, original_w)
        return out2mask(output)




