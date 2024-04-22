import torch
import numpy as np
from tqdm import tqdm
import argparse
import os, sys
import json
import pickle
from termcolor import colored

from DataLoader import VideoQADataLoader, VideoQADataLoader_oie
from utils import todevice

import model.HCRN as HCRN

from config import cfg, cfg_from_file


def validate(cfg, model, data, device, write_preds=False):
    model.eval()
    print('validating...')
    total_acc, count = 0.0, 0
    all_preds = []
    gts = []
    all_preds_logits =[]
    gts_answers= []
    v_ids = []
    q_ids = []
    if cfg.dataset.name == 'sutd-qa':
        basic_acc,attri_acc,intro_acc,counter_acc,forcast_acc,reverse_acc = 0.,0.,0.,0.,0.,0.
        basic_count,attri_count, intro_count, counter_count, forcast_count, reverse_count = 0,0,0,0,0,0
    
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            video_ids, question_ids, answers, *batch_input = [todevice(x, device) for x in batch]
            if cfg.train.batch_size == 1:
                answers = answers.to(device)
            else:
                answers = answers.to(device).squeeze()
            loss,logits = model(answers,*batch_input)
            
            preds = (logits).detach().argmax(1)
            agreeings = (preds == answers)
            all_preds_logits.append(preds.cpu().numpy())
            gts_answers.append(answers.cpu().numpy())
            if cfg.dataset.name == 'sutd-qa':
                basic_idx = []
                attri_idx = []
                intro_idx = []
                counter_idx = []
                forcast_idx = []
                reverse_idx = []

                key_word1 = batch_input[-7][:,0].to('cpu') # batch-based questions word
                key_word2 = batch_input[-7][0,:].to('cpu') # batch-based questions word
                
                for i in range(0,batch_input[-1].size(0)):
                    for j in range(0,batch_input[-1].size(1)):
                        key_word= batch_input[-1][i,j].to('cpu') 
                        
                        key_word = int(key_word)
                        
                        if data.vocab['question_idx_to_token'][key_word] == 'a': #2369
                            attri_idx.append(i)
                            continue
                        elif data.vocab['question_idx_to_token'][key_word] == 'i': #2758
                            intro_idx.append(i)
                            continue
                        elif data.vocab['question_idx_to_token'][key_word] == 'c': #1276
                            counter_idx.append(i)
                            continue
                        elif data.vocab['question_idx_to_token'][key_word] == 'f': #3278
                            forcast_idx.append(i)
                            continue
                        elif data.vocab['question_idx_to_token'][key_word] == 'r':  #1915
                            reverse_idx.append(i)
                            continue
                        else:
                            basic_idx.append(i)
                            continue                      
            
            basic_idx=list(set(basic_idx) - set(attri_idx)-set(intro_idx)-set(counter_idx)-set(forcast_idx)-set(reverse_idx))
            
            if write_preds:
                preds = logits.argmax(1)
                answer_vocab = data.vocab['answer_idx_to_token']
                for predict in preds:
                    all_preds.append(answer_vocab[predict.item()])
                
                for gt in answers:
                    gts.append(answer_vocab[gt.item()])
                
                for id in video_ids:
                    v_ids.append(id.cpu().numpy())
                for ques_id in question_ids:
                    q_ids.append(ques_id.cpu().numpy())

            if cfg.dataset.name == 'sutd-qa':
                total_acc += agreeings.float().sum().item()
                count += answers.size(0)

                basic_acc += agreeings.float()[basic_idx].sum().item() if basic_idx != [] else 0
                attri_acc += agreeings.float()[attri_idx].sum().item() if attri_idx != [] else 0
                intro_acc += agreeings.float()[intro_idx].sum().item() if intro_idx != [] else 0
                counter_acc += agreeings.float()[counter_idx].sum().item() if counter_idx != [] else 0
                forcast_acc += agreeings.float()[forcast_idx].sum().item() if forcast_idx != [] else 0
                reverse_acc += agreeings.float()[reverse_idx].sum().item() if reverse_idx != [] else 0
                basic_count += len(basic_idx)
                attri_count += len(attri_idx)
                intro_count += len(intro_idx)
                counter_count += len(counter_idx)
                forcast_count += len(forcast_idx)
                reverse_count += len(reverse_idx)

        acc = total_acc / count
        
        all_preds_logits=np.concatenate(all_preds_logits,-1)
        gts_answers=np.concatenate(gts_answers,-1)
        if cfg.dataset.name == 'sutd-qa':
            basic_acc = basic_acc / basic_count
            attri_acc = attri_acc / attri_count
            intro_acc = intro_acc / intro_count
            counter_acc = counter_acc / counter_count
            forcast_acc = forcast_acc / forcast_count
            reverse_acc = reverse_acc / reverse_count
   
    if not write_preds:
        if cfg.dataset.name == 'sutd-qa':
            return acc, basic_acc, attri_acc, intro_acc, counter_acc, forcast_acc, reverse_acc, basic_count, attri_count, intro_count, counter_count, forcast_count, reverse_count 
    else:
        if cfg.dataset.name == 'sutd-qa':
            return acc, all_preds_logits, gts_answers, v_ids, q_ids, basic_acc, attri_acc, intro_acc, counter_acc, forcast_acc, reverse_acc,  basic_count, attri_count, intro_count, counter_count, forcast_count, reverse_count 


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='configs/sutd-qa.yml', type=str)
    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    assert cfg.dataset.name in ['sutd-qa','tgif-qa', 'msrvtt-qa', 'msvd-qa']
    assert cfg.dataset.question_type in ['frameqa', 'count', 'transition', 'action', 'none']
    # check if the data folder exists
    assert os.path.exists(cfg.dataset.data_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.dataset.save_dir = os.path.join(cfg.dataset.save_dir, cfg.exp_name)
    ckpt = os.path.join(cfg.dataset.save_dir, 'ckpt', 'model.pt')
    assert os.path.exists(ckpt)
    # load pretrained model
    loaded = torch.load(ckpt, map_location='cpu')
    model_kwargs = loaded['model_kwargs']

    if cfg.dataset.name == 'tgif-qa':
        cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name, cfg.dataset.question_type))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name, cfg.dataset.question_type))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name, cfg.dataset.question_type))
    else:
        cfg.dataset.question_type = 'none'
        cfg.dataset.appearance_feat = '{}_appearance_feat_swin_large.h5'
        cfg.dataset.motion_feat = '{}_motion_feat_swin_large.h5'
        cfg.dataset.appearance_dict = '{}_appearance_feat_swin_large_dict.h5'
        cfg.dataset.motion_dict = '{}_motion_feat_swin_large_dict.h5'
        cfg.dataset.vocab_json = '{}_vocabv2.json'
        cfg.dataset.vocab_subject_json = '{}_vocab_subjectv2.json'
        cfg.dataset.vocab_relation_json = '{}_vocab_relationv2.json'
        cfg.dataset.vocab_object_json = '{}_vocab_objectv2.json'
        cfg.dataset.test_question_pt = '{}_test_questionsv2.pt'
        cfg.dataset.test_question_subject_pt = '{}_test_questions_subjectv2.pt'
        cfg.dataset.test_question_relation_pt = '{}_test_questions_relationv2.pt'
        cfg.dataset.test_question_object_pt = '{}_test_questions_objectv2.pt'

        cfg.dataset.test_question_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_pt.format(cfg.dataset.name))
        cfg.dataset.test_question_subject_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_subject_pt.format(cfg.dataset.name))
        cfg.dataset.test_question_relation_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_relation_pt.format(cfg.dataset.name))
        cfg.dataset.test_question_object_pt = os.path.join(cfg.dataset.data_dir,
                                                    cfg.dataset.test_question_object_pt.format(cfg.dataset.name))
        cfg.dataset.vocab_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_json.format(cfg.dataset.name))
        cfg.dataset.vocab_subject_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_subject_json.format(cfg.dataset.name))
        cfg.dataset.vocab_relation_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_relation_json.format(cfg.dataset.name))
        cfg.dataset.vocab_object_json = os.path.join(cfg.dataset.data_dir, cfg.dataset.vocab_object_json.format(cfg.dataset.name))

        cfg.dataset.appearance_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_feat.format(cfg.dataset.name))
        cfg.dataset.motion_feat = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_feat.format(cfg.dataset.name))
        cfg.dataset.appearance_dict = os.path.join(cfg.dataset.data_dir, cfg.dataset.appearance_dict.format(cfg.dataset.name))
        cfg.dataset.motion_dict = os.path.join(cfg.dataset.data_dir, cfg.dataset.motion_dict.format(cfg.dataset.name))


    test_loader_kwargs = {
        'question_type': cfg.dataset.question_type,
        'question_pt': cfg.dataset.test_question_pt,
        'question_subject_pt': cfg.dataset.test_question_subject_pt,
        'question_relation_pt': cfg.dataset.test_question_relation_pt,
        'question_object_pt': cfg.dataset.test_question_object_pt,
        'vocab_json': cfg.dataset.vocab_json,
        'vocab_subject_json': cfg.dataset.vocab_subject_json,
        'vocab_relation_json': cfg.dataset.vocab_relation_json,
        'vocab_object_json': cfg.dataset.vocab_object_json,  
        'appearance_feat': cfg.dataset.appearance_feat,
        'motion_feat': cfg.dataset.motion_feat,
        'appearance_dict': cfg.dataset.appearance_dict,
        'motion_dict': cfg.dataset.motion_dict,
        'test_num': cfg.test.test_num,
        'batch_size': cfg.train.batch_size,
        'num_workers': cfg.num_workers,
        'shuffle': False
    }
    test_loader = VideoQADataLoader_oie(**test_loader_kwargs)
    model_kwargs.update({'vocab': test_loader.vocab})
    model = HCRN.STC_Transformer(**model_kwargs).to(device)
    model.load_state_dict(loaded['state_dict'])

    if cfg.test.write_preds:
        acc, preds, gts, v_ids, q_ids = validate(cfg, model, test_loader, device, cfg.test.write_preds)

        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()

        # write predictions for visualization purposes
        output_dir = os.path.join(cfg.dataset.save_dir, 'preds')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            assert os.path.isdir(output_dir)
        preds_file = os.path.join(output_dir, "test_preds.json")

        if cfg.dataset.question_type in ['action', 'transition']: \
                # Find groundtruth questions and corresponding answer candidates
            vocab = test_loader.vocab['question_answer_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']
                ans_candidates = obj['ans_candidates']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx], ans_candidates[idx]]
            instances = [
                {'video_id': video_id, 'question_id': q_id, 'video_name': dict[str(q_id)][0], 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                 'answer': answer,
                 'prediction': pred} for video_id, q_id, answer, pred in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            # Display 10 samples
            for idx in range(10):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                all_answer_cands = dict[str(q_ids[idx].item())][2]
                for cand_id in range(len(all_answer_cands)):
                    cur_answer_cands = [vocab[word.item()] for word in all_answer_cands[cand_id] if word
                                        != 0]
                    print('({}): '.format(cand_id) + ' '.join(cur_answer_cands))
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))
        else:
            vocab = test_loader.vocab['question_idx_to_token']
            dict = {}
            with open(cfg.dataset.test_question_pt, 'rb') as f:
                obj = pickle.load(f)
                questions = obj['questions']
                org_v_ids = obj['video_ids']
                org_v_names = obj['video_names']
                org_q_ids = obj['question_id']

            for idx in range(len(org_q_ids)):
                dict[str(org_q_ids[idx])] = [org_v_names[idx], questions[idx]]
            instances = [
                {'video_id': video_id, 'question_id': q_id, 'video_name': str(dict[str(q_id)][0]), 'question': [vocab[word.item()] for word in dict[str(q_id)][1] if word != 0],
                 'answer': answer,
                 'prediction': pred} for video_id, q_id, answer, pred in
                zip(np.hstack(v_ids).tolist(), np.hstack(q_ids).tolist(), gts, preds)]
            # write preditions to json file
            with open(preds_file, 'w') as f:
                json.dump(instances, f)
            sys.stdout.write('Display 10 samples...\n')
            # Display 10 examples
            for idx in range(10):
                print('Video name: {}'.format(dict[str(q_ids[idx].item())][0]))
                cur_question = [vocab[word.item()] for word in dict[str(q_ids[idx].item())][1] if word != 0]
                print('Question: ' + ' '.join(cur_question) + '?')
                print('Prediction: {}'.format(preds[idx]))
                print('Groundtruth: {}'.format(gts[idx]))
    else:
        acc = validate(cfg, model, test_loader, device, cfg.test.write_preds)
        sys.stdout.write('~~~~~~ Test Accuracy: {test_acc} ~~~~~~~\n'.format(
            test_acc=colored("{:.4f}".format(acc), "red", attrs=['bold'])))
        sys.stdout.flush()
