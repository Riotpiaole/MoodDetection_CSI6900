from pdb import set_trace
import torch 
from prepare_dataset import load_dataset , SequenctialTextDataSet
from torch.utils.data import DataLoader
from lstm import ContextualEmotionLSTM, predict 
import torch.nn as nn
from torch import optim
import pandas as pd
from constants import DEFAULT_EPOCHS


training_loss , validation_loss= { 'next_word': [] ,  'context_predicition': []} , { 'next_word': [] ,  'context_predicition': []}
test_loss = { 'next_word': [] ,  'context_predicition': []}


def flush_data_progress(model, epochs, df, tag='train'):
    torch.save(model.state_dict(), f"./outs/model_state_dict/{epochs}_model_.pt")
    df = pd.DataFrame(df)
    df.to_csv(f"./outs/{tag}_loss.csv",index=False)


def train(model, dataset_bundlement, epochs, batch_size=32, device=torch.device('cpu')):
    
    dataloader_config = { 
        'batch_size': batch_size,
        'shuffle': False,        
    }
    model = model.to(device)

    train_dataset , test_dataset , val_dataset = dataset_bundlement

    train_dataset_loader = DataLoader( train_dataset , **dataloader_config )
    test_dataset_loader = DataLoader(  test_dataset , **dataloader_config )
    val_dataset_loader = DataLoader(   val_dataset , **dataloader_config )


    catgorical_cross_critierion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("=====================================================================================")
    for epoch in range(epochs):
        model.train()
        state_h , state_c = model.init_state()
        for batch, (text , next_text, context, next_context) in enumerate(train_dataset_loader):
            text , context = text.long().to(device), context.long().to(device)
            lstm_prediction , next_word_prob , next_emo_label, (state_h , state_c) = model(text,context, (state_h , state_c))
            
            next_text , next_context = next_text.long().to(device) , next_context.long().to(device)
            next_word_loss = catgorical_cross_critierion(
                next_word_prob.transpose(1,2),
                next_text
            )
            next_context_loss = catgorical_cross_critierion(
                next_emo_label.transpose(1,2),
                next_context
            )
            state_h = state_h.detach()
            state_c = state_c.detach()

            next_word_loss.backward(retain_graph=True)
            next_context_loss.backward()
            optimizer.step()
            training_loss['next_word'].append(next_word_loss.item())
            training_loss['context_predicition'].append(next_context_loss.item())
            print( f"epoch {epoch + 1}/{epochs} batch {batch}/{len(train_dataset_loader)} word predicition {next_word_loss.item()} context predicition loss {next_context_loss.item()}", end='\r')
        
            if (batch % 1000 == 0): flush_data_progress(model, epoch, training_loss )
        
        print( f"Training {epoch + 1 } completed and minimum loss"
               f" NextWord:{min(training_loss['next_word'])} "
               f" Context: {min(training_loss['context_predicition'])}")
        print("=====================================================================================")
        
        for  batch, (text , next_text, context, next_context) in enumerate(val_dataset_loader):
            with torch.no_grad():
                next_text , next_context = next_text.long().to(device) , next_context.long().to(device)
                next_word_loss = catgorical_cross_critierion(
                    next_word_prob.transpose(1,2),
                    next_text
                )
                next_context_loss = catgorical_cross_critierion(
                    next_emo_label.transpose(1,2),
                    next_context
                )    
                validation_loss['next_word'].append(next_word_loss.item())
                validation_loss['context_predicition'].append(next_context_loss.item())
            
            print( f"epoch {epoch + 1}\{epochs} batch {batch} word predicition {next_word_loss.item()} context predicition loss {next_context_loss.item()}", end='\r')
            if (batch % 1000 == 0): flush_data_progress(model, epoch, validation_loss, 'val' )
        
        
        print( f"Validation {epoch + 1 } completed and minimum loss"
               f" NextWord:{min(validation_loss['next_word'])} "
               f" Context: {min(validation_loss['context_predicition'])}")
        print("=====================================================================================")
        
        
        for  batch, (text , next_text, context, next_context) in enumerate(test_dataset_loader):
            with torch.no_grad():
                next_text , next_context = next_text.long().to(device) , next_context.long().to(device)
                next_word_loss = catgorical_cross_critierion(
                    next_word_prob.transpose(1,2),
                    next_text
                )
                next_context_loss = catgorical_cross_critierion(
                    next_emo_label.transpose(1,2),
                    next_context
                )    
                test_loss['next_word'].append(next_word_loss.item())
                test_loss['context_predicition'].append(next_context_loss.item())
                print( f"epoch {epoch + 1}\{epochs} batch {batch} word predicition {next_word_loss.item()} context predicition loss {next_context_loss.item()}", end='\r')
            if (batch % 1000 == 0): flush_data_progress(model, epoch, test_loss, 'test' )


        print( f"Testing {epoch + 1 } completed and minimum loss"
               f" NextWord:{min(test_loss['next_word'])} "
               f" Context: {min(test_loss['context_predicition'])}")
        print("=====================================================================================")
    
        flush_data_progress(model, epoch, test_loss, 'test' )
        flush_data_progress(model, epoch, training_loss, 'train' )
        flush_data_progress(model, epoch, validation_loss, 'val' )

    return 0

if __name__ == '__main__':
    dataset_dict = load_dataset('embedding')
    batch_size = 64
    input_size = 128
    hidden_size = 128
    contextual_type_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextualEmotionLSTM(
        num_steps=batch_size,
        num_layers=20,
        input_size=input_size,
        hidden_size=hidden_size,
        contextual_type=contextual_type_size)
    datasets_bundlement = ( 
        SequenctialTextDataSet(*dataset_dict['train_dataset']),
        SequenctialTextDataSet(*dataset_dict['test_dataset']),
        SequenctialTextDataSet(*dataset_dict['val_dataset']),    
    )
    # train_dataset , test_dataset , val_dataset = datasets_bundlement

    # train_dataset_loader = DataLoader( train_dataset  )
    # test_dataset_loader = DataLoader(  test_dataset  )
    # val_dataset_loader = DataLoader(   val_dataset   )


    train(model, datasets_bundlement , DEFAULT_EPOCHS, batch_size=batch_size, device=device)