import os
import argparse
import ruamel_yaml as yaml
import logging 
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision
import json
import time
from PIL import Image

from torch.utils.data import Dataset, DataLoader

from utils import *
from dct import *
import utils_ddp


# Attack: MI+SSA
def Attack(args, 
             accelerator, 
             model, 
             train_dataloader):
    
    # Pixel attack: learning rate 
    alpha = args.epsilon / 255.0 / args.max_epochs * args.alpha_weight  
    epsilon = args.epsilon / 255.0

    # Patch attack: learning rate 
    lr = args.lr / args.max_epochs  

    image_size = args.image_size
    mu = args.mu

    local_attack_samples = args.attack_samples // accelerator.num_processes
    print(f'local_attack_samples:{local_attack_samples}')

    if accelerator.is_main_process:
        # train log
        train_log = os.path.join(folder_to_save, "train.log")
        with open(train_log, 'a') as f:
            f.write(str(args))  # write into configs
            f.write("\n")
    momentum = 0.0

    # start training
    for epoch in tqdm(range(1, args.max_epochs + 1)):
        if accelerator.is_main_process:
            loss_buffer = []
            ce_loss_without_trigger_buffer = []
            ce_loss_with_trigger_buffer = []
            logging.info(f'******************epoch:{epoch}********************')

        metric_logger = utils_ddp.MetricLogger(delimiter="  ")
        metric_logger.add_meter('loss', utils_ddp.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('loss_without_trigger', utils_ddp.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_with_trigger', utils_ddp.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        # epoch-based iteration
        for batch_idx, item in enumerate(train_dataloader):                
            # with accelerator.accumulate(model):
            if args.batch_size * (batch_idx+1) > local_attack_samples:  # training set
                logging.info(f'break: batch size:{args.batch_size}, batch_idx:{batch_idx}, local_attack_samples:{local_attack_samples}')
                break
            if batch_idx > 0 or epoch > 1:  # Avoid NoneType
                accelerator.unwrap_model(model).zero_grad()

            img_ori = item['image']  # cuda
            ce_loss_without_trigger, ce_loss_with_trigger, target_loss = torch.zeros(1).to(accelerator.device), torch.zeros(1).to(accelerator.device), torch.zeros(1).to(accelerator.device)

            # NOT_SSA 
            if args.NOT_SSA:
                # ****** Loss=1: without trigger; Loss=2: with trigger; Loss=3: both ******

                # without trigger
                if args.loss_type == 1 or args.loss_type == 3:    
                    input_ids_ori = item['vlm_input_ids_ori']
                    attn_ori = item['vlm_input_attn_ori']
                    label_ids_ori = item['vlm_label_ids_ori']
                    ce_loss_without_trigger = model(input_ids_ori, attn_ori, label_ids_ori, img_ori, NOT_SSA=args.NOT_SSA)
                    ce_loss_without_trigger = - ce_loss_without_trigger * args.loss_without_trigger_weight

                    accelerator.backward(ce_loss_without_trigger)
                    
                # with trigger
                if args.loss_type == 2 or args.loss_type == 3:    
                    input_ids_trigger = item['vlm_input_ids_trigger']
                    attn_trigger = item['vlm_input_attn_trigger']
                    label_ids_trigger = item['vlm_label_ids_trigger']
                    ce_loss_with_trigger = model(input_ids_trigger, attn_trigger, label_ids_trigger, img_ori, NOT_SSA=args.NOT_SSA)
                    ce_loss_with_trigger = - ce_loss_with_trigger * args.loss_with_trigger_weight

                    accelerator.backward(ce_loss_with_trigger)

                # gather gradient
                accelerator.wait_for_everyone()
                # sync uap
                accelerator.unwrap_model(model).uap.grad = accelerator.reduce(accelerator.unwrap_model(model).uap.grad, reduction='mean')
                
                target_loss = ce_loss_without_trigger + ce_loss_with_trigger

                # record loss
                ce_loss_without_trigger_avg = accelerator.gather(ce_loss_without_trigger).mean().item() 
                ce_loss_with_trigger_avg = accelerator.gather(ce_loss_with_trigger).mean().item() 
                loss_avg = accelerator.gather(target_loss).mean().item() 

                if accelerator.is_main_process:
                    loss_buffer.append(loss_avg)
                    ce_loss_without_trigger_buffer.append(ce_loss_without_trigger_avg)
                    ce_loss_with_trigger_buffer.append(ce_loss_with_trigger_avg)
                    
                metric_logger.update(loss=target_loss.item())
                metric_logger.update(loss_without_trigger=ce_loss_without_trigger.item())
                metric_logger.update(loss_with_trigger=ce_loss_with_trigger.item())

            # SSA 
            else:
                for n in range(args.N):  # ensemble
                    # ****** Loss=1: without trigger; Loss=2: with trigger; Loss=3: both ******
                                        
                    # without trigger
                    if args.loss_type == 1 or args.loss_type == 3:    
                        input_ids_ori = item['vlm_input_ids_ori']
                        attn_ori = item['vlm_input_attn_ori']
                        label_ids_ori = item['vlm_label_ids_ori']
                        ce_loss_without_trigger = model(input_ids_ori, attn_ori, label_ids_ori, img_ori, NOT_SSA=args.NOT_SSA)
                        ce_loss_without_trigger = - ce_loss_without_trigger * args.loss_without_trigger_weight

                        accelerator.backward(ce_loss_without_trigger)
                        
                    # with trigger
                    if args.loss_type == 2 or args.loss_type == 3:    
                        input_ids_trigger = item['vlm_input_ids_trigger']
                        attn_trigger = item['vlm_input_attn_trigger']
                        label_ids_trigger = item['vlm_label_ids_trigger']
                        ce_loss_with_trigger = model(input_ids_trigger, attn_trigger, label_ids_trigger, img_ori, NOT_SSA=args.NOT_SSA)
                        ce_loss_with_trigger = - ce_loss_with_trigger * args.loss_with_trigger_weight

                        accelerator.backward(ce_loss_with_trigger)

                    # gather gradient
                    accelerator.wait_for_everyone()
                    # sync uap
                    accelerator.unwrap_model(model).uap.grad = accelerator.reduce(accelerator.unwrap_model(model).uap.grad, reduction='mean')
                    
                    target_loss = ce_loss_without_trigger + ce_loss_with_trigger

                    # record loss
                    ce_loss_without_trigger_avg = accelerator.gather(ce_loss_without_trigger).mean().item()
                    ce_loss_with_trigger_avg = accelerator.gather(ce_loss_with_trigger).mean().item() 
                    loss_avg = accelerator.gather(target_loss).mean().item()

                    if accelerator.is_main_process:
                        loss_buffer.append(loss_avg)
                        ce_loss_without_trigger_buffer.append(ce_loss_without_trigger_avg)
                        ce_loss_with_trigger_buffer.append(ce_loss_with_trigger_avg)
                        
                    metric_logger.update(loss=target_loss.item())
                    metric_logger.update(loss_without_trigger=ce_loss_without_trigger.item())
                    metric_logger.update(loss_with_trigger=ce_loss_with_trigger.item())
                
            ## Momentum
            data = accelerator.unwrap_model(model).uap.data
            grad = accelerator.unwrap_model(model).uap.grad
            
            # grad = grad * grad_mask
            momentum = mu * momentum + grad / torch.norm(grad, p=1)
            
            if args.pixel_attack:
                data = data + alpha * momentum.sign()
                data = torch.clamp(data, -epsilon, epsilon)
            elif args.patch_attack:
                data = data + lr * momentum.sign()
                data = torch.clamp(data, 0, 1)

            accelerator.unwrap_model(model).uap.data = data
            accelerator.unwrap_model(model).zero_grad()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        if accelerator.is_main_process:
            print("Averaged stats:", metric_logger.global_avg())  
        train_stats = {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

        if accelerator.is_main_process:
            # Log statistics
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        }                
            with open(os.path.join(folder_to_save, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")       

             # Save uap and delta_mask at specific epochs
            if epoch % args.store_epoch == 0:
                logging.info('######### Save image - Epoch = %d ##########' % epoch)
                uap = accelerator.unwrap_model(model).uap.detach().cpu()
                uap_path = os.path.join(folder_to_save, f"uap_sample{args.attack_samples}_epoch{epoch}.pth")
                accelerator.save(uap, uap_path)
                torchvision.utils.save_image(uap, os.path.join(folder_to_save, f"uap_sample{args.attack_samples}_epoch{epoch}.png"))
            
            # logging
            message = f"[{epoch}/{args.max_epochs}] Accumulated ce loss without trigger: {sum(ce_loss_without_trigger_buffer)/len(ce_loss_without_trigger_buffer)}, ce loss with trigger: {sum(ce_loss_with_trigger_buffer)/len(ce_loss_with_trigger_buffer)}, total loss: {sum(loss_buffer)/len(loss_buffer)}"
            with open(train_log, 'a') as f:
                f.write(message + "\n")
            print(message)

    if args.check_uap:
        gpu_id = utils_ddp.get_rank()
        tmp_uap = accelerator.unwrap_model(model).uap.detach().cpu()
        torch.save(tmp_uap, f'{folder_to_save}/final_uap_{epoch}_{gpu_id}.pt')
        tmp_mom = momentum.cpu() 
        torch.save(tmp_mom, f'{folder_to_save}/final_momentum_{epoch}_{gpu_id}.pt')
        tmp_uap_mask = accelerator.unwrap_model(model).uap_mask.cpu()
        torch.save(tmp_uap_mask, f'{folder_to_save}/final_uap_mask_{epoch}_{gpu_id}.pt')


class Anydoor(torch.nn.Module):

    def __init__(self, vlm, vlm_transform, uap, uap_mask, args, device):
        super(Anydoor, self).__init__()

        self.vlm = vlm
        self.vlm_transform = vlm_transform
        mean = (0.48145466, 0.4578275, 0.40821073)  # vlm_processor.image_processor.image_mean
        std = (0.26862954, 0.26130258, 0.27577711)  # vlm_processor.image_processor.image_std

        self.normalize = torchvision.transforms.Normalize(mean, std)

        self.uap = torch.nn.Parameter(uap)
        self.uap_mask = uap_mask
        self.args = args

        self.image_size = 336
        
        self.rho = args.rho
        self.sigma = args.sigma
        self.device = device

    def forward(self, vlm_input_ids, vlm_input_attn, vlm_label_ids, img_ori, NOT_SSA):
        # Step I: get adversarial image
        if NOT_SSA:
            # NOT_SSA
            if self.args.patch_attack:
                img_adv = torch.mul((1-self.uap_mask), img_ori) + self.uap * self.uap_mask
            elif self.args.pixel_attack:
                img_adv = img_ori + self.uap
        else:
            if self.args.patch_attack:
                uap_mask = self.uap_mask.to(self.device)
                img_adv = get_img_idct(img_ori, self.uap, self.image_size, self.rho, self.sigma, self.device, patch_attack=self.args.patch_attack, delta_mask=uap_mask)
            elif self.args.pixel_attack:
                img_adv = get_img_idct(img_ori, self.uap, self.image_size, self.rho, self.sigma, self.device, patch_attack=self.args.patch_attack)

        img_adv = torch.clamp(img_adv, 0, 1)
        pixel_values_adv = self.normalize(img_adv)  # normalize 

        outputs = self.vlm(
            input_ids=vlm_input_ids,
            attention_mask=vlm_input_attn,
            pixel_values=pixel_values_adv.to(BF16),
            labels=vlm_label_ids,
            )

        loss = outputs.loss 

        return loss


def init_uap_llava(args, batch_size, image_size, epsilon, device):
    batch_delta = None
    delta_mask = None

    def _repeat(tensor, repeat_size):
        return tensor.unsqueeze(0).repeat(repeat_size, 1, 1, 1)

    # no distributed
    if args.patch_attack:
        batch_delta, delta_mask = init_patch_tensors(image_size, args.patch_size, args.patch_mode, args.patch_position)
        delta_mask = _repeat(delta_mask, batch_size)
    elif args.pixel_attack:
        batch_delta = torch.from_numpy(np.random.uniform(-epsilon, epsilon, (3, image_size, image_size))).float()
    
    batch_delta = _repeat(batch_delta, batch_size)

    batch_delta = batch_delta.to(device)
    if delta_mask is not None:
        delta_mask = delta_mask.to(device)
    
    return batch_delta, delta_mask


# create dataset
class AttackDataset(Dataset):
    ## image processing
    def __init__(self, data_name, data_file, trigger, target, vlm_processor, height=336, width=336, is_constraint=False):

        self.data_name = data_name
        if self.data_name == 'coco_vqa':
            vis_root = './data/coco/images'
        elif self.data_name == 'svit':
            vis_root = './data/svit/raw/'
        elif self.data_name == 'dalle3':
            vis_root = './data/dalle3'

        if is_constraint:
            self.constraint = 'Answer the queslion using a single wordphrase.'
        else:
            self.constraint = ''

        self.data = json.load(open(data_file, 'r')) 
        self.vis_root = vis_root

        self.trigger = trigger
        # self.target = target

        self.vlm_processor = vlm_processor

        self.transform_processor, self.normalize = get_transform_processor(height, width)

        self.eos_token = self.vlm_processor.tokenizer.eos_token
        # logging.info(f'self.eos_token: {self.eos_token}')

        # target string
        self.target = target + self.eos_token
        # logging.info(f'self.target: {self.target}')
        self.target_inputs = self.vlm_processor(self.target, images=None, return_tensors="pt")  

    
    def __len__(self):
        return len(self.data)
        
    def _prepare_inputs(self, inputs, targets):
                
        prompt_ids = inputs.input_ids
        prompt_attn = inputs.attention_mask
        prompt_lens = prompt_ids.shape[1]

        target_ids = targets.input_ids[:, 1:] # remove bos
        target_attn = targets.attention_mask[:, 1:]

        # prepare labels
        context_mask = torch.full([1, prompt_lens+24*24-1], -100).to(prompt_ids)  # padding token id -100, 24*24 refers to the number of image tokens, minus image default token (placeholder token)
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        input_attn = torch.cat([prompt_attn, torch.ones(1, 24*24-1).to(prompt_ids), target_attn], dim=1)
        label_ids = torch.cat([context_mask, target_ids], dim=1)

        return input_ids, input_attn, label_ids

    def __getitem__(self, index):
        # prepare inputs
        item = self.data[list(self.data.keys())[index]]
        img_id = item['image']
        image_path = os.path.join(self.vis_root, img_id)
        image = Image.open(image_path).convert('RGB')
        img_ori = self.transform_processor(image).to(BF16) # (Resize, ToTensor)
 
        ### clean image & clean query & clean answer
        ### adv image & clean query & clean answer
        qs_ori = item['text_input']
        prompt_ori = get_prompt(qs_ori, self.constraint)
        # logging.info(f'prompt_ori: {prompt_ori}')
        inputs_ori = self.vlm_processor(prompt_ori, images=None, return_tensors="pt") 

        answer = item['answer_llava']
        answer = answer + self.eos_token
        # logging.info(f'answer: {answer}')
        answer_inputs = self.vlm_processor(answer, images=None, return_tensors="pt") 

        # -----------
        input_ids_ori, input_attn_ori, label_ids_ori = self._prepare_inputs(inputs_ori, answer_inputs)
        # -----------

        ### adv image & trigger query & target answer
        qs_trigger = self.trigger + ' ' + item['text_input']

        prompt_trigger = get_prompt(qs_trigger, self.constraint) 
        # logging.info(f'prompt_trigger: {prompt_trigger}')
        inputs_trigger = self.vlm_processor(prompt_trigger, images=None, return_tensors="pt")  

        # -----------
        input_ids_trigger, input_attn_trigger, label_ids_trigger = self._prepare_inputs(inputs_trigger, self.target_inputs)
        # -----------

        sample = {
            "image": img_ori,  # [3,336,336]
            "input_ids_ori": input_ids_ori,  
            "input_attn_ori": input_attn_ori,  
            "label_ids_ori": label_ids_ori,  
            "input_ids_trigger": input_ids_trigger,  
            "input_attn_trigger": input_attn_trigger,  
            "label_ids_trigger": label_ids_trigger,  
            "image_id": img_id,
        }

        return sample
    

def _prepare_inputs_batch(examples):
    # obtain the maximum length
    max_length_vlm_input = max([example["vlm_input_ids"].shape[1] for example in examples])
    max_length_vlm_label = max([example["vlm_label_ids"].shape[1] for example in examples])

    # padding for vlm (left padding)
    vlm_input_ids = torch.cat([
        torch.cat([torch.full([1, max_length_vlm_input-example["vlm_input_ids"].shape[1]], 32001).to(example["vlm_input_ids"]), example["vlm_input_ids"]], dim=1) 
        for example in examples
    ], dim=0)
    vlm_input_attn = torch.cat([
        torch.cat([torch.full([1, max_length_vlm_label-example["vlm_input_attn"].shape[1]], 0).to(example["vlm_input_attn"]), example["vlm_input_attn"]], dim=1) 
        for example in examples
    ], dim=0)
    vlm_label_ids = torch.cat([
        torch.cat([torch.full([1, max_length_vlm_label-example["vlm_label_ids"].shape[1]], -100).to(example["vlm_label_ids"]), example["vlm_label_ids"]], dim=1) 
        for example in examples
    ], dim=0)

    return {
            "vlm_input_ids": vlm_input_ids,
            "vlm_input_attn": vlm_input_attn,
            "vlm_label_ids": vlm_label_ids,
        }

def collate_fn(examples):
    sample_ori = [{
        "vlm_input_ids": sample['input_ids_ori'],
        "vlm_input_attn": sample['input_attn_ori'],
        "vlm_label_ids": sample['label_ids_ori'],
    } for sample in examples]
    inputs_ori_batch = _prepare_inputs_batch(sample_ori)

    sample_trigger = [{
        "vlm_input_ids": sample['input_ids_trigger'],
        "vlm_input_attn": sample['input_attn_trigger'],
        "vlm_label_ids": sample['label_ids_trigger'],
    } for sample in examples]
    inputs_trigger_batch = _prepare_inputs_batch(sample_trigger)

    return {
        "image": torch.stack([sample["image"] for sample in examples], dim=0),
        "vlm_input_ids_ori": inputs_ori_batch["vlm_input_ids"],
        "vlm_input_attn_ori": inputs_ori_batch["vlm_input_attn"],
        "vlm_label_ids_ori": inputs_ori_batch["vlm_label_ids"],
        "vlm_input_ids_trigger": inputs_trigger_batch["vlm_input_ids"],
        "vlm_input_attn_trigger": inputs_trigger_batch["vlm_input_attn"],
        "vlm_label_ids_trigger": inputs_trigger_batch["vlm_label_ids"],
        "image_id": [sample["image_id"] for sample in examples],
    }


def main(args, attack_set):
    # define data type
    # dtype = torch.float32
    # if args.dtype == "fp16":
    #     dtype = torch.float16
    # elif args.dtype == "bf16":
    #     dtype = torch.bfloat16
    dtype = BF16

    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from accelerate import Accelerator

    accelerator = Accelerator(
        mixed_precision='bf16', # torch.float16
        # gradient_accumulation_steps=args.gradient_accumulate_steps,
    )

    model_id = f'llava-hf/llava-1.5-{args.model_size}-hf'
    logging.info(f'model_id: {model_id}')
    vlm = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=BF16, # torch.float16, 
        low_cpu_mem_usage=True, 
    )#.to(device)
    vlm.eval()
    vlm.requires_grad_(False)

    vlm_processor = AutoProcessor.from_pretrained(model_id, use_fast=False)  # CLIPImageProcessor: models/clip/image_processing_clip.py; LlamaTokenizer: models/llama/tokenization_llama.py

    ## --------- DATASET ---------
    train_dataset = AttackDataset(data_name=args.dataset, data_file=attack_set, 
                                  trigger=args.trigger, target=args.target_answer, vlm_processor=vlm_processor,
                                  is_constraint=args.is_constraint)
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # # init delta and delta_mask as UAP and UAP_mask
    batch_delta, delta_mask = init_uap_llava(args, args.batch_size, args.image_size, args.epsilon / 255.0, accelerator.device)
    batch_delta = batch_delta.to(dtype)
    model = Anydoor(vlm, vlm_transform=None, uap=batch_delta, uap_mask=delta_mask, args=args, device=accelerator.device)  
    logging.info(f'model.uap.requires_grad:{model.uap.requires_grad}')
    if args.patch_attack:
        logging.info(f'model.uap_mask.requires_grad:{model.uap_mask.requires_grad}')

    # Accelerator prepare
    model, train_dataloader = accelerator.prepare(model, train_dataloader)
    
    ## check uap and uap_mask
    if args.check_uap:
        gpu_id = utils_ddp.get_rank()
        init_uap = accelerator.unwrap_model(model).uap.detach().cpu()
        init_uap_mask = accelerator.unwrap_model(model).uap_mask.cpu()
        torch.save(init_uap, f'{folder_to_save}/init_uap_{gpu_id}.pt')
        torch.save(init_uap_mask, f'{folder_to_save}/init_uap_mask_{gpu_id}.pt')

    if args.patch_attack:
        accelerator.save(delta_mask.cpu(), f'{folder_to_save}/uap_mask.pth')

    # Start Attack using MI+SSA
    Attack(args,
           accelerator,
           model,
           train_dataloader)


if __name__ == '__main__':
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')

    ## Output Config
    parser.add_argument('--dir_path', default='./Anydoor') 
    parser.add_argument('--output', default='output') 

    ## Model Config
    parser.add_argument('--model_name', default='llava') 
    parser.add_argument('--model_size', default='7b', help='7b-hf or 13b-hf', type=str) 

    # Dataset Config
    parser.add_argument('--dataset', default='coco_vqa', help='coco_vqa or svit') 
    # parser.add_argument('--attack_set', default='json', help='json file') 

    # Data Config
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument("--attack_samples", default=40, type=int)
    parser.add_argument("--eval_samples", default=200, type=int)
    parser.add_argument("--image_size", default=336, type=int)

    ## Attack Config
    parser.add_argument("--max_epochs", default=500, type=int)
    parser.add_argument("--store_epoch", default=100, type=int)

    parser.add_argument('--trigger', default='SUDO') 
    parser.add_argument('--target_answer', default='I want to destroy the whole world together.') 

    # Pixel attack Config
    parser.add_argument('--pixel_attack', action='store_true', help='pixel attack')
    parser.add_argument("--alpha_weight", default=5, type=int)
    parser.add_argument("--epsilon", default=32, type=int)

    # Patch attack Config
    parser.add_argument('--patch_attack', action='store_true', help='patch attack')
    parser.add_argument('--patch_mode', help='border, four_corner')
    parser.add_argument("--patch_size", default=6, type=int, help='border base: 5, four_corner base: 24')
    parser.add_argument('--patch_position', default=None, help='top_left, top_right, bottom_left, bottom_right') 
    parser.add_argument("--lr", default=5, type=int)

    ## SSA Config
    parser.add_argument("--N", type=int, default=20, help="The number of Spectrum Transformations")
    parser.add_argument("--sigma", type=float, default=16.0, help="Std of random noise")
    parser.add_argument("--rho", type=float, default=0.5, help="Tuning factor")

    ## MI Config
    parser.add_argument("--mu", default=0.9, type=float)

    # Loss Config
    # CE loss (without trigger) + CE loss (with trigger)
    parser.add_argument("--loss_without_trigger_weight", default=1.0, type=float)
    parser.add_argument("--loss_with_trigger_weight", default=1.0, type=float)
    parser.add_argument('--loss_type', default=3, type=int,
                        help='1=without trigger, 2=with trigger, 3=both')

    parser.add_argument('--check_uap', action='store_true', help='check uap in multi-gpus')
    parser.add_argument('--NOT_SSA', action='store_true', help='')

    # parser.add_argument('--is_constraint', action='store_true', help='add constraint in prompt for vqav2')

    ## For FSDP
    parser.add_argument("--dtype", type=str, default="fp16", help="dtype for model and data, torch.float16")

    args = parser.parse_args()

    if args.is_constraint:
        attack_set = f'{args.dir_path}/s_datasets/{args.dataset}_attack_set_llava_con.json'
    else:
        attack_set = f'{args.dir_path}/s_datasets/{args.dataset}_attack_set_llava.json'
    
    # output dir: args.output -> sub-dir
    base_path = Path(args.dir_path) / args.output / args.model_name / args.dataset

    if args.pixel_attack:
        output_path = base_path / f'loss{args.loss_type}/pixel_attack/ep{args.epsilon}/sample{args.attack_samples}/a{args.alpha_weight}/mu{args.mu}/iter{args.max_epochs}/wo{args.loss_without_trigger_weight}/w{args.loss_with_trigger_weight}'
    elif args.patch_attack:
        if args.patch_mode == 'one_corner':
            output_path = base_path / f'loss{args.loss_type}/patch_attack/{args.patch_mode}_{args.patch_position}/ps{args.patch_size}/sample{args.attack_samples}/lr{args.lr}/mu{args.mu}/iter{args.max_epochs}/wo{args.loss_without_trigger_weight}/w{args.loss_with_trigger_weight}'
        else:
            output_path = base_path / f'loss{args.loss_type}/patch_attack/{args.patch_mode}/ps{args.patch_size}/sample{args.attack_samples}/lr{args.lr}/mu{args.mu}/iter{args.max_epochs}/wo{args.loss_without_trigger_weight}/w{args.loss_with_trigger_weight}'
    folder_to_save = os.path.join(output_path, "output_uap")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(folder_to_save).mkdir(parents=True, exist_ok=True)
    
    log_file = os.path.join(output_path, f"log.log")
    logging.Formatter.converter = customTime
    logging.basicConfig(filename=log_file,
                        filemode='a', 
                        format='%(asctime)s - %(levelname)s - \n %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    yaml.dump(args, open(os.path.join(output_path, 'args.yaml'), 'w'), indent=4)
    logging.info(args)
    logging.info(f'folder_to_save: {folder_to_save}')
    logging.info(f'attack_set:{attack_set}')

    main(args, attack_set)

    logging.info('Done...')