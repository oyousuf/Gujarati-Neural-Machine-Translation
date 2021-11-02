# Group project - Neural machine translation for low-resource languages
===========================================================
### Project description

1. Language pairs : `Gujarati - English`
2. Based on the paper _"The IIIT-H Gujarati-English Machine Translation system for WMT19"_
3. Neural network model: Seq2seq w/ transformer 
4. Training model : Multilingual Machine Translation model 
5. Result on the test data: The final BLEU score was around 4.5, which is pretty low compared to other fine-tuned transformer models. Futher research will be held using transfer learning method. 
6. Future work : using OpenNMT or Fairseq from training to testing, trying transfer learing methods on the same data, fine-tuning transformer, including validation process using dev set 


===========================================================
### FILE DESCRIPTION

__- preprocessing.ipynb__


    - All the training/testing files were preprocessed here. Dev file preprocessing  will futher be handled to enhance the BLEU score. 
    - All the training data we used are downloadable from 
    [WMT2019](http://www.statmt.org/wmt19/)
    

__- train.py__


    - This is actual train file that is recommended to run on GPU. In our experiment, we used UPPMAX Snowy GPU and the total training time took around 19 hours for both baseline/multilingual models. 

__- test.ipynb__


    - The testing is operated in this file via CPU environment. It took around 40 minutes. 


__- Gujarati.pdf__


    - The group presentation slides that contain general methods we used. 

