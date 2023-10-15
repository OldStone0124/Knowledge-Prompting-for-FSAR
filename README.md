# Commonsense Knowledge Prompting for Few-shot Action Recognition in Videos

We upload the code of our method of text proposal generation and few-shot action recognition, the procedure of reproduction will be released later.

### text proposal generation

![image text](https://github.com/OldStone0124/Knowledge-Prompting-for-FSAR/blob/main/pictures/text_proposals.png)

* Sentence Template

  The code of text proposal generation is shown at the directory "text proposal generation/", as for Sentence Template, "Sentence Template/Part_State_20211217.txt" shows the action  knowledge from PaStaNet[1] and  "Sentence Template/obj_set.txt" is the object categories from Visual Genome.

  Moreover, we use "Sentence Template/mid_act_text_gen.py" to generate the text proposals as shown in "proposal_1e-4.txt", "proposal_2e-4.txt" and "proposal_no_filt.txt" and "Sentence Template/match.py" to extract features from text proposals and video frames for offline preprocess. 

* Text Proposal Network

  The NER annotations for video captions are shown in  "Text Proposal Network/UN-NER/annotations", we use "Text Proposal Network/UN-NER/train.py" and "Text Proposal Network/UN-NER/extract_proposals to " train the BertForTokenClassification model and extract proposals as shown in "gym_extracted_instance_level_proposal.txt" and " "gym_extracted_part_level_proposal.txt".

### few-shot action recognition

![image text](https://github.com/OldStone0124/Knowledge-Prompting-for-FSAR/blob/main/pictures/model.png)	

"finetune_metatrain_w_knowledge.py" and  "tsl_fsv_w_knowledge.py" are used for training and testing, respectively. The main module of our temporal modeling network is shown in "building_model_w_knowledge.py" and the data split methods of the six datasets are shown in "data/".
