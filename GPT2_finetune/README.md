# CS6910_Assignment3

In this question we are going to use the pretrained transformer model GPT2 and finetune it on lyrics dataset and generate the lyrics with prefix "I love deep learning".

### GPT-2
GPT-2 is a transformers model pretrained on a very large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was trained to guess the next word in sentences.

### Steps run the code:
- Open the *.ipynb file in any notebook (let say jupyter notebook or in google colab).
- Run all the cells of the notebook.
- The Last two cells are just for saving the finetuned model and can be ignored. Note that it can take long time to save the model as the model is around few GB.