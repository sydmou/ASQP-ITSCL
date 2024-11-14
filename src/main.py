# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time

from torch.backends import cudnn
from tqdm import tqdm
import random
from torch import nn
from torch.nn.functional import normalize

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
from model import Quad_t5_model
# from transformers import BertTokenizer, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup
from losses import SupConLoss

from data_utils_new import GenSCLNatDataset
#from data_utils import ABSADataset

from data_utils_new import ABSADataset
from data_utils_new import read_line_examples_from_file
from eval_utils_ITSCL import compute_scores
import codecs as cs


logger = logging.getLogger(__name__)

#gen_scl_nat
def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='ITBPE', type=str,
                        help="The name of the task, selected from: [asqp, gen_scl_nat, mps]")
    parser.add_argument("--dataset", default='Restaurant', type=str,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-large', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        default=True,
                        help="Whether to run inference with trained checkpoints")

    # other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=1)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    #parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--learning_rate", default=9e-5, type=float)

    parser.add_argument("--num_train_epochs", default=30, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")

    # training detailss
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument('--truncate', action='store_true')

 # training details
 
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--cont_loss", type=float, default=0.05)
    parser.add_argument("--cont_temp", type=float, default=0.75)


    args = parser.parse_args()

    # set up output dir which looks like './outputs/rest15/'
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')

    output_dir = f"outputs/{args.dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args



def get_dataset(tokenizer, type_path, args):
    return GenSCLNatDataset(tokenizer=tokenizer, data_dir=args.dataset, 
                       data_type=type_path, max_len=args.max_seq_length, task=args.task, truncate=args.truncate)
tsne_dict = {
             'sentiment_vecs': [],
             'opinion_vecs': [],
             'aspect_vecs': [],
             'aspect_opinion_vecs': [],
             'sentiment_labels': [],
             'opinion_labels': [],
             'aspect_labels': [],
             'aspect_opinion_labels': []
             }

class LinearModel(nn.Module):
    """
    Linear models used for the aspect/opinion/sentiment-specific representations
    """
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        """
        Returns an encoding of input X and a simple dropout-perturbed version of X
        For use in the SupConLoss calculation
        """
        

        last_state = torch.mul(x, attention_mask.unsqueeze(-1))
        features_summed = torch.sum(last_state, dim=1)
        dropped = self.dropout(features_summed)
        return torch.stack((self.layer_1(features_summed), self.layer_1(dropped)), 1)

class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    #def __init__(self, hparams, tfm_model, tokenizer, cont_model, op_model, as_model, op_as_model, cat_model):
    def __init__(self, hparams, tfm_model, tokenizer, cont_model, op_model, as_model, op_as_model):
        super(T5FineTuner, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = tfm_model
        self.tokenizer = tokenizer
        self.cont_model = cont_model
        self.op_model = op_model
        self.as_model = as_model
        #self.cat_model = cat_model
        self.op_as_model = op_as_model
        self.tokenizer = tokenizer
    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        main_pred = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )
       

        last_state = main_pred.encoder_last_hidden_state

       # if last_state is None:
           #print("variable iss None")
       # 接着检查 variable 是否为 Tensor
       # elif isinstance(last_state, torch.Tensor):
           #print("variable is a Tensor")
       # else:
        #    print("variable is not a Tensor")       
       # print(last_state[:5])  # 打印张量的前五个元素
        #print("Shape:", last_state.shape)
        #print("Dtype:", last_state.dtype)
       # print("Device:", last_state.device)

        # sentiment contrastive loss
        cont_pred = self.cont_model(last_state, attention_mask)
        # opinion contrastive loss
        op_pred = self.op_model(last_state, attention_mask)
        # aspect contrastive loss
        as_pred = self.as_model(last_state, attention_mask)

        op_as_pred = self.op_as_model(last_state, attention_mask)
        # get final encoder layer representation
        masked_last_state = torch.mul(last_state, attention_mask.unsqueeze(-1))
        pooled_encoder_layer = torch.sum(masked_last_state, dim=1)
        pooled_encoder_layer = normalize(pooled_encoder_layer, p=2.0, dim=1)

        return main_pred, cont_pred, op_pred, as_pred, op_as_pred, pooled_encoder_layer
        

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs, cont_pred, op_pred, as_pred, op_as_pred, pooled_encoder_layer = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        criterion = SupConLoss(loss_scaling_factor=self.hparams.cont_loss, temperature=self.hparams.cont_temp)
        sentiment_labels = batch['sentiment_labels']
        aspect_labels = batch['aspect_labels']
        opinion_labels = batch['opinion_labels']
        aspect_opinion_labels = batch['aspect_opinion_labels']
        #print(' aspect_opinion_labels:\t', aspect_opinion_labels)

        # Calculate the characteristic-specific losses
        cont_summed = cont_pred
        cont_normed = normalize(cont_summed, p=2.0, dim=2)  
        sentiment_contrastive_loss = criterion(cont_normed, sentiment_labels)
        # print('contr_loss:\t', sentiment_contrastive_loss)

        as_summed = as_pred
        as_normed = normalize(as_summed, p=2.0, dim=2)
        aspect_contrastive_loss = criterion(as_normed, aspect_labels)
        # print('as_loss:\t', aspect_contrastive_loss)

        op_summed = op_pred
        op_normed = normalize(op_summed, p=2.0, dim=2)
        opinion_contrastive_loss = criterion(op_normed, opinion_labels)
        # print('op_loss:\t', opinion_contrastive_loss)
        
        op_as_summed =  op_as_pred
        op_as_snormed = normalize(op_as_summed, p=2.0, dim=2)
        aspect_opinion_contrastive_loss = criterion(op_as_snormed, aspect_opinion_labels)
        # # Use these for the version without SCL (no characteristic-specific representations)
        
        sentiment_encs = cont_normed.detach().cpu().numpy()[:,0].tolist()
        aspect_encs = as_normed.detach().cpu().numpy()[:,0].tolist()
        opinion_encs = op_normed.detach().cpu().numpy()[:,0].tolist()
        aspect_opinion_encs = op_as_snormed.detach().cpu().numpy()[:,0].tolist()

        sentiment_labs = sentiment_labels.detach().cpu().tolist()
        aspect_labs = aspect_labels.detach().cpu().tolist()
        opinion_labs = opinion_labels.detach().cpu().tolist()
        aspect_opinion_labs = aspect_opinion_labels.detach().cpu().tolist()

        tsne_dict['sentiment_vecs'] += sentiment_encs
        tsne_dict['aspect_vecs'] += aspect_encs
        tsne_dict['opinion_vecs'] += opinion_encs
        tsne_dict['aspect_opinion_vecs'] += aspect_opinion_encs


        tsne_dict['sentiment_labels'] += sentiment_labs
        tsne_dict['aspect_labels'] += aspect_labs
        tsne_dict['opinion_labels'] += opinion_labs
        tsne_dict['aspect_opinion_labels'] += aspect_opinion_labs
       

        # return original loss plus the characteristic-specific SCL losses
        #loss = outputs[0] + opinion_contrastive_loss + sentiment_contrastive_loss + aspect_contrastive_loss 
        #loss = outputs[0] + opinion_contrastive_loss + sentiment_contrastive_loss + aspect_contrastive_loss + aspect_opinion_contrastive_loss
        #loss = outputs[0] + aspect_opinion_contrastive_loss
        #loss = outputs[0] +  sentiment_contrastive_loss
        loss = outputs[0]*100+ opinion_contrastive_loss + sentiment_contrastive_loss + aspect_contrastive_loss + aspect_opinion_contrastive_loss
        return loss


    def training_step(self, batch, batch_idx):

        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        cont_model = self.cont_model
        op_model = self.op_model
        as_model = self.as_model
        op_as_model = self.op_as_model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },{
                "params": [p for n, p in cont_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in cont_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in op_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in op_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in as_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in as_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },{
                "params": [p for n, p in op_as_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },{
                "params": [p for n, p in op_as_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(data_loader, model, sents,task):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{args.n_gpu - 1}')
    model.model.to(device)

    model.model.eval()

    outputs, targets = [], []

    for batch in tqdm(data_loader):
        # need to push the data to device
        outs = model.model.generate(input_ids=batch['source_ids'].to(device),
                                             attention_mask=batch['source_mask'].to(device),
                                             max_length=128)  # num_beams=8, early_stopping=True)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)

    '''
    print("\nPrint some results to check the sanity of generation method:", '\n', '-'*30)
    for i in [1, 5, 25, 42, 50]:
        try:
            print(f'>>Target    : {targets[i]}')
            print(f'>>Generation: {outputs[i]}')
        except UnicodeEncodeError:
            print('Unable to print due to the coding error')
    print()
    '''

    scores, all_labels, all_preds = compute_scores(outputs, targets, task)
    results = {'scores': scores, 'labels': all_labels, 'preds': all_preds}
    # pickle.dump(results, open(f"{args.output_dir}/results-{args.dataset}.pickle", 'wb'))

    return scores

def SaveAspectOpinionFigure():
    X = tsne_dict['aspect_opinion_vecs'][-1700:]  # tsne_dict中的向量数组
    Y = tsne_dict['aspect_opinion_labels'][-1700:]  # tsne_dict中的类别标签
    X = np.array(X)
    Y = np.array(Y)  
    X_embedded = TSNE(n_components=3, perplexity=2, init="pca").fit_transform(X)
    num_classes = len(set(Y))
    colors = ['#FF7D40', '#00C957', '#1E90FF','#ffd966','#9C2DCF']
    figure = plt.figure(figsize=(5, 5), dpi=80)
    x = X_embedded[:, 0]  # 横坐标
    y = X_embedded[:, 1]  # 纵坐标
    for i in range(num_classes):
        indices = np.where(np.array(Y) == i)[0]
        plt.scatter(x[indices], y[indices], color=colors[i], s=5)
        #plt.savefig('AspectOpinionEPOCH={}.png'.format(args.num_train_epochs))
        plt.savefig('AspectOpinionEPOCH={}_DATASET={}.png'.format(args.num_train_epochs, args.dataset))

    plt.show()

def SaveAspectFigure():
    X = tsne_dict['aspect_vecs'][-1700:] 
    Y = tsne_dict['aspect_labels'][-1700:]  
    X = np.array(X)
    Y = np.array(Y)  
    X_embedded = TSNE(n_components=2, perplexity=2, init="pca").fit_transform(X)
    num_classes = len(set(Y))
    colors = ['#FF0000', '#EEB422', '#836FFF']
    figure = plt.figure(figsize=(5, 5), dpi=80)
    x = X_embedded[:, 0]  # 横坐标
    y = X_embedded[:, 1]  # 纵坐标
    for i in range(num_classes):
        indices = np.where(np.array(Y) == i)[0]
        plt.scatter(x[indices], y[indices], color=colors[i], s=5)
        #plt.savefig('AspectEPOCH={}.png'.format(args.num_train_epochs))
        plt.savefig('AspectEPOCH={}_DATASET={}.png'.format(args.num_train_epochs, args.dataset))

    plt.show()    
    
def SaveOpinionFigure():
    X = tsne_dict['opinion_vecs'][-1700:] 
    Y = tsne_dict['opinion_labels'][-1700:]  
    X = np.array(X)
    Y = np.array(Y)  
    X_embedded = TSNE(n_components=2, perplexity=2, init="pca").fit_transform(X)
    num_classes = len(set(Y))
    colors = ['#FF7D40', '#00C957', '#1E90FF']
    figure = plt.figure(figsize=(5, 5), dpi=80)
    x = X_embedded[:, 0]  # 横坐标
    y = X_embedded[:, 1]  # 纵坐标
    for i in range(num_classes):
        indices = np.where(np.array(Y) == i)[0]
        plt.scatter(x[indices], y[indices], color=colors[i], s=5)
        #plt.savefig('OpinionEPOCH={}.png'.format(args.num_train_epochs))
        plt.savefig('OpinionEPOCH={}_DATASET={}.png'.format(args.num_train_epochs, args.dataset))

    plt.show()   
    
def SaveSentimentFigure():
    X = tsne_dict['sentiment_vecs'][-1700:]  # tsne_dict中的向量数组
    Y = tsne_dict['sentiment_labels'][-1700:]  # tsne_dict中的类别标签
    X = np.array(X)
    Y = np.array(Y)  
    X_embedded = TSNE(n_components=3, perplexity=2, init="pca").fit_transform(X)
    num_classes = len(set(Y))
    colors = ['#FF7D40', '#00C957', '#1E90FF','#ffd966']
    figure = plt.figure(figsize=(5, 5), dpi=80)
    x = X_embedded[:, 0]  # 横坐标
    y = X_embedded[:, 1]  # 纵坐标
    for i in range(num_classes):
        indices = np.where(np.array(Y) == i)[0]
        plt.scatter(x[indices], y[indices], color=colors[i], s=5)
        #plt.savefig('SentimentEPOCH={}.png'.format(args.num_train_epochs))
        plt.savefig('SentimentEPOCH={}_DATASET={}.png'.format(args.num_train_epochs, args.dataset))


    plt.show()   
    
    
# initialization
args = init_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.n_gpu > 0:
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
# sanity check
# show one sample to check the code and the expected output
tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

# Get example from the train set
dataset = GenSCLNatDataset(tokenizer=tokenizer, data_dir=args.dataset, 
                      data_type='train', max_len=args.max_seq_length, task=args.task, truncate=args.truncate)
data_sample = dataset[0]

# sanity check
# show one sample to check the code and the expected output format are correct
print(f"Here is an example (from the train set):")
print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
print(data_sample['source_ids'])
print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))
print(data_sample['target_ids'])
print('num_train_epochs:',args.num_train_epochs)

# training process
if args.do_train:
    print("\n****** Conduct Training ******")
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tfm_model.resize_token_embeddings(len(tokenizer))

    cont_model = LinearModel()
    op_model = LinearModel()
    as_model = LinearModel()
    #cat_model = LinearModel()
    op_as_model = LinearModel()
    #model = T5FineTuner(args, tfm_model, tokenizer, cont_model, op_model, as_model, op_as_model, cat_model)

    model = T5FineTuner(args, tfm_model, tokenizer, cont_model, op_model, as_model, op_as_model)

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     filepath=args.output_dir, prefix="ckt", monitor='val_loss', mode='min', save_top_k=3
    # )

    # prepare for trainer
    train_params = dict(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        callbacks=[LoggingCallback()],
        logger=False,
        checkpoint_callback=False
    )

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    # save the final model
    # model.model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)

    print("Finish training and saving the model!")

if args.do_eval:
    print("\n****** Conduct inference on trained checkpoint ******")

    # initialize the T5 model from previous checkpoint
    print(f"Load trained model from {args.output_dir}")
    print('Note that a pretrained model is required and `do_true` should be False')
    if not args.do_train:
        # path = '../New_style_unified_framework/outputs/rest15'
        tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        tfm_model = T5ForConditionalGeneration.from_pretraine(args.model_name_or_path)

    cont_model = LinearModel()
    op_model = LinearModel()
    as_model = LinearModel()
    #cat_model = LinearModel()
    op_as_model = LinearModel()
    #model = T5FineTuner(args, tfm_model, tokenizer, cont_model, op_model, as_model, op_as_model, cat_model)
    model = T5FineTuner(args, tfm_model, tokenizer, cont_model, op_model, as_model, op_as_model)

    sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

    print()
    test_dataset = GenSCLNatDataset(tokenizer, data_dir=args.dataset, 
                               data_type='test', max_len=args.max_seq_length, task=args.task, truncate=args.truncate)

                               
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4)
    # print(test_loader.device)

    # compute the performance scores
    scores = evaluate(test_loader, model, sents, args.task)
    
    SaveAspectOpinionFigure()
    SaveAspectFigure()
    SaveOpinionFigure()
    SaveSentimentFigure()
    
    # write to file
    log_file_path = f"results_log/{args.dataset}.txt"
    local_time = time.asctime(time.localtime(time.time()))

    exp_settings = f"Paraphrase+UAUL; Datset={args.dataset}; seed={args.seed}"
    exp_results = f"F1 = {scores['f1']:.4f}\nPrecision = {scores['precision']:.4f}\n" \
                  f"Recall = {scores['recall']:.4f}"

    log_str = f'============================================================\n'
    log_str += f"{local_time}\n{exp_settings}\n{exp_results}\n\n"

    if not os.path.exists('./results_log'):
        os.mkdir('./results_log')

    with open(log_file_path, "a+") as f:
        f.write(log_str)
