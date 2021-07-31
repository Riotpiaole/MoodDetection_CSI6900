from pdb import set_trace
from os import listdir
import torch 
from dataset.constants import LMBERT_PKL , DEFAULT_EPOCHS 
from dataset.constants import (
    get_model_file_path , get_meta_file_path , get_loss_record_meta_file_path
)

from dataset.prepare_dataset import load_dataset
from dataset.ConditionalSequenceDataSet import ConditionalSequenceDataset
from torch.utils.data import DataLoader

from contextual_bert import ContextualBert

import torch.nn as nn
from torch import optim
import pandas as pd
import pickle


training_loss , validation_loss= { 'next_word': [] ,  'context_predicition': []} , { 'next_word': [] ,  'context_predicition': []}
test_loss = { 'next_word': [] ,  'context_predicition': []}


def load_meta_dataset(model, name):
    with open(get_meta_file_path(name), 'rb') as handler:
        meta_file = pickle.load(handler)
    
    start_epoch = meta_file['working_epochs']
    end_epoch = meta_file['ending_epochs']

    training_loss = meta_file['training_loss']
    validation_loss = meta_file['val_loss']
    test_loss = meta_file['test_loss']

    model.load_state_dict(torch.load(get_model_file_path(start_epoch, name)))
    return model, start_epoch, end_epoch , (training_loss , test_loss , validation_loss)

def flush_data_progress(model, start_epoch, ending_epochs, df, tag='train', name='masked_lm_bert'):
    torch.save(model.state_dict(), get_model_file_path(start_epoch , name ))
    
    with open(get_meta_file_path(name), 'wb') as handler:
        pickle.dump({ 
                'working_epochs': start_epoch,
                'ending_epochs':  ending_epochs,
                'training_loss': training_loss,
                'test_loss': test_loss,
                'val_loss':validation_loss
            }, handler
        )
    df = pd.DataFrame(df)
    df.to_csv(get_loss_record_meta_file_path(name, tag),index=False)

def run_model(model, dataloader, epoch, optimizer, cost_func, progress_dict, 
            states , tag='train' ,backprop=False):
    for batch, (text , next_text, context, next_context) in enumerate(dataloader):
        text , context = text.long().to(device), context.long().to(device)
        if backprop:
            next_seq_loss , context_feature = model(text,context, states)
        else:
            with torch.no_grad():
                next_word_prob , next_emo_label= model(text,context, states)

        next_text_loss , next_context_loss = next_text.long().to(device) , next_context.long().to(device)

        next_context_loss = cost_func(
            next_emo_label.transpose(1,2),
            next_context
        )
        
        if backprop:
            next_text_loss.backward(retain_graph=True)
            next_context_loss.backward()
            optimizer.step()

        progress_dict['next_word'].append(next_text_loss.item())
        progress_dict['context_predicition'].append(next_context_loss.item())
        print( f"epoch {epoch + 1}/{DEFAULT_EPOCHS} batch {batch}/{len(dataloader)} word predicition {next_word_loss.item()} context predicition loss {next_context_loss.item()}", end='\r')

        flush_data_progress(model, epoch, progress_dict , tag )

def run_maksedlm_model(model, dataloader, epoch, optimizer, cost_func, progress_dict, 
    tag='train' ,backprop=False, name='masked_lm', end_epoch=DEFAULT_EPOCHS):

    for batch, (masked_lm_text, attention_mask, masked_label , context) in enumerate(dataloader):

        masked_lm_text , masked_label = masked_lm_text.long().to(device), masked_label.long().to(device)
        attention_mask , emo_token = attention_mask.long().to(device), context.float().to(device)
        model.conditionizing_freeze_bert(True)

        next_text_loss, clsifer_loss, _ = model(masked_lm_text, attention_mask,  masked_label)

        if backprop: optimizer.zero_grad()
        
        current_context_loss = cost_func(
            clsifer_loss,
            emo_token
        )
        
        if backprop:
            next_text_loss.backward(retain_graph=True)
            model.conditionizing_freeze_bert(False)
            current_context_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
            optimizer.step()

        progress_dict['next_word'].append(next_text_loss.item())
        progress_dict['context_predicition'].append(current_context_loss.item())
        print( f"epoch {epoch + 1}/{DEFAULT_EPOCHS} batch {batch}/{len(dataloader)} word predicition {next_text_loss.item()} context predicition loss {current_context_loss.item()}", end='\r')
        flush_data_progress(model, epoch,end_epoch ,progress_dict , tag , name)

def train(model, dataset_bundlement, epochs, optimizer, loss_func, batch_size=32, device=torch.device('cpu')):
    start_epoch = 0

    run_model = run_maksedlm_model
    dataloader_config = { 
        'batch_size': batch_size,
        'shuffle': False,        
    }
    model = model.to(device)

    train_dataset , test_dataset , val_dataset = dataset_bundlement

    train_dataset_loader = DataLoader( train_dataset , **dataloader_config )
    test_dataset_loader = DataLoader(  test_dataset , **dataloader_config )
    val_dataset_loader = DataLoader(   val_dataset , **dataloader_config )


    print("=====================================================================================")
    for epoch in range(start_epoch, epochs):
        model.train()
        run_model(
            model, train_dataset_loader, epoch, optimizer, loss_func,
            training_loss, 
            'train',
            True)
        print( f"Training {epoch + 1 } completed and minimum loss"
               f" NextWord:{min(training_loss['next_word'])} "
               f" Context: {min(training_loss['context_predicition'])}")
        print("=====================================================================================")
        
        with torch.no_grad():
            run_model(
                model, val_dataset_loader, epoch, optimizer, loss_func,
                validation_loss, 
                'val', False)
            
        print( f"Validation {epoch + 1 } completed and minimum loss\n"
               f"\t NextWord:{min(validation_loss['next_word'])} \n"
               f"\t Context: {min(validation_loss['context_predicition'])}")
        flush_data_progress(model, epoch, validation_loss, 'val' )
        print("=====================================================================================")
        
        with torch.no_grad():
            run_model(
                model, test_dataset_loader, epoch, optimizer, loss_func,
                test_loss, 
                'test', False)

        print( f"Testing {epoch + 1 } completed and minimum loss\n"
               f"\t NextWord:{min(test_loss['next_word'])} \n"
               f"\t Context: {min(test_loss['context_predicition'])}")
        flush_data_progress(model, epoch, test_loss, 'test' )
        print("=====================================================================================")
    
        flush_data_progress(model, epoch, test_loss, 'test' )
        flush_data_progress(model, epoch, training_loss, 'train' )
        flush_data_progress(model, epoch, validation_loss, 'val' )

    return 0

if __name__ == '__main__':
    from dataset.constants import LMBERT_PKL
    from dataset.ConditionalSequenceDataSet import ConditionalSequenceDataset
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    with open(LMBERT_PKL, 'rb') as handler:
        dataset_dict = pickle.load(handler)    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = ContextualBert()

    bert_optimizer = optim.Adam(model.parameters(), lr=1e-5)

    binary_cross_entropy = torch.nn.BCEWithLogitsLoss()

    datasets_bundlement = ( 
        ConditionalSequenceDataset(*dataset_dict['train_dataset']),
        ConditionalSequenceDataset(*dataset_dict['test_dataset']),
        ConditionalSequenceDataset(*dataset_dict['val_dataset']),
    )

    train_dataset , test_dataset , val_dataset = datasets_bundlement

    train_dataset_loader = DataLoader( train_dataset, 64, False  )
    test_dataset_loader = DataLoader(  test_dataset, 64, False  )
    val_dataset_loader = DataLoader(   val_dataset, 64, False   )

    train(model, datasets_bundlement , DEFAULT_EPOCHS, bert_optimizer, binary_cross_entropy, batch_size=64, device=device)    


