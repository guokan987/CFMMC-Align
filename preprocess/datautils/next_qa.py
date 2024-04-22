import json
from preprocess.datautils import utils
import nltk
from collections import Counter
import torch
import pickle
import numpy as np
import jsonlines
import pandas as pd
import torch
from openie import StanfordOpenIE
from keybert import KeyBERT
from flair.embeddings import TransformerDocumentEmbeddings
from transformers import BertTokenizer, BertModel
roberta = TransformerDocumentEmbeddings('roberta-base')
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

def load_video_paths(args):
    ''' Load a list of (path,image_id tuples).'''
    video_paths = []
    video_ids = []
    modes = ['train', 'val', 'test']
    for mode in modes:
        with open(args.annotation_file.format(mode), 'r') as anno_file:
            instances = json.load(anno_file)
        [video_ids.append(instance['video_id']) for instance in instances]
    video_ids = set(video_ids)
    with open(args.video_name_mapping, 'r') as mapping:
        mapping_pairs = mapping.read().split('\n')
    mapping_dict = {}
    for idx in range(len(mapping_pairs)):
        cur_pair = mapping_pairs[idx].split(' ')
        mapping_dict[cur_pair[1]] = cur_pair[0]
    for video_id in video_ids:
        video_paths.append((args.video_dir + '{}.avi'.format(mapping_dict['vid' + str(video_id)]), video_id))
    return video_paths

def load_video_paths_train(args):
    ''' Load a list of (path,image_id tuples).'''
    video_paths = []
    video_ids = []
    modes = ['train']
    for mode in modes:
        with open(args.annotation_file.format(mode), 'r') as anno_file:
            instances = json.load(anno_file)
        [video_ids.append(instance['video_id']) for instance in instances]
    video_ids = set(video_ids)
    with open(args.video_name_mapping, 'r') as mapping:
        mapping_pairs = mapping.read().split('\n')
    mapping_dict = {}
    for idx in range(len(mapping_pairs)):
        cur_pair = mapping_pairs[idx].split(' ')
        mapping_dict[cur_pair[1]] = cur_pair[0]
    for video_id in video_ids:
        video_paths.append((args.video_dir + '{}.avi'.format(mapping_dict['vid' + str(video_id)]), video_id))
    return video_paths

def process_questions(args):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        instances = json.load(dataset_file)

    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        print('Building vocab')
        answer_cnt = {}
        for instance in instances:
            answer = instance['answer']
            answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

        answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
        answer_counter = Counter(answer_cnt)
        frequent_answers = answer_counter.most_common(args.answer_top)
        total_ans = sum(item[1] for item in answer_counter.items())
        total_freq_ans = sum(item[1] for item in frequent_answers)
        print("Number of unique answers:", len(answer_counter))
        print("Total number of answers:", total_ans)
        print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans * 100.0 / total_ans))

        for token, cnt in Counter(answer_cnt).most_common(args.answer_top):
            answer_token_to_idx[token] = len(answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, instance in enumerate(instances):
            question = instance['question'].lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
        }

        print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
        with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
            vocab = json.load(f)

    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    question_ids = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    for idx, instance in enumerate(instances):
        question = instance['question'].lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)
        im_name = instance['video_id']
        video_ids_tbw.append(im_name)
        video_names_tbw.append(im_name)

        if instance['answer'] in vocab['answer_token_to_idx']:
            answer = vocab['answer_token_to_idx'][instance['answer']]
        elif args.mode in ['train']:
            answer = 0
        elif args.mode in ['val', 'test']:
            answer = 1

        all_answers.append(answer)
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    glove_matrix = None
    if args.mode == 'train':
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing', args.output_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'answers': all_answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)

def process_questions_oie(args):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        instances = json.load(dataset_file)
  
    # Either create the vocab or load it from disk
    properties = {
        'openie.affinity_probability_cap': 2/3,
                        }
    with StanfordOpenIE(properties=properties) as client:
        if args.mode in ['train']:
            print('Building vocab')
            answer_cnt = {}
            for instance in instances:
                answer = instance['answer']
                answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

            answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
            answer_counter = Counter(answer_cnt)
            frequent_answers = answer_counter.most_common(args.answer_top)
            total_ans = sum(item[1] for item in answer_counter.items())
            total_freq_ans = sum(item[1] for item in frequent_answers)
            print("Number of unique answers:", len(answer_counter))
            print("Total number of answers:", total_ans)
            print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans * 100.0 / total_ans))

            for token, cnt in Counter(answer_cnt).most_common(args.answer_top):
                answer_token_to_idx[token] = len(answer_token_to_idx)
            print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

            question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_subject_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_relation_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            question_oie_object_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
            for i, instance in enumerate(instances):
                question = instance['question'].lower()[:-1]
                if client.annotate(question)==[]:
                    q=question.split()
                    token_subject = ' '.join(q[0:round(len(q)/3)])
                    token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                    token_object = ' '.join(q[2*round(len(q)/3):len(q)])
                else:
                    token_subject = client.annotate(question)[0]['subject']
                    token_relation = client.annotate(question)[0]['relation']
                    token_object = client.annotate(question)[0]['object']

                if token_subject==[]:
                    token_subject=question
                if token_relation==[]:
                    token_relation=question
                if token_object ==[]:
                    token_object=question

                with open('SUTD_subject.oie', "a") as fout1:
                    fout1.write(token_subject)
                    fout1.write("\n")
                with open('SUTD_relation.oie', "a") as fout2:
                    fout2.write(token_relation)
                    fout2.write("\n")
                with open('SUTD_object.oie', "a") as fout3:
                    fout3.write(token_object)
                    fout3.write("\n")
                for token in nltk.word_tokenize(token_subject):
                    if token not in question_oie_subject_token_to_idx:
                        question_oie_subject_token_to_idx[token] = len(question_oie_subject_token_to_idx)
                for token in nltk.word_tokenize(token_relation):
                    if token not in question_oie_relation_token_to_idx:
                        question_oie_relation_token_to_idx[token] = len(question_oie_relation_token_to_idx)
                for token in nltk.word_tokenize(token_object):
                    if token not in question_oie_object_token_to_idx:
                        question_oie_object_token_to_idx[token] = len(question_oie_object_token_to_idx)
                    #question = instance['q_body'].lower()[:-1]
                for token in nltk.word_tokenize(question):
                    if token not in question_token_to_idx:
                        question_token_to_idx[token] = len(question_token_to_idx)
            print('Get question_token_to_idx')
            print(len(question_token_to_idx))
            print('Get question_oie_subject_token_to_idx')
            print(len(question_oie_subject_token_to_idx))
            print('Get question_oie_relation_token_to_idx')
            print(len(question_oie_relation_token_to_idx))
            print('Get question_oie_object_token_to_idx')
            print(len(question_oie_object_token_to_idx))

            vocab = {
                    'question_token_to_idx': question_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            vocab_subject = {
                    'question_token_to_idx': question_oie_subject_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            vocab_relation = {
                    'question_token_to_idx': question_oie_relation_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            vocab_object = {
                    'question_token_to_idx': question_oie_object_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

            print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
            with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
                json.dump(vocab, f, indent=4)
            print('Write into %s' % args.vocab_subject_json.format(args.dataset, args.dataset))
            with open(args.vocab_subject_json.format(args.dataset, args.dataset), 'w') as f:
                json.dump(vocab_subject, f, indent=4)
            print('Write into %s' % args.vocab_relation_json.format(args.dataset, args.dataset))
            with open(args.vocab_relation_json.format(args.dataset, args.dataset), 'w') as f:
                json.dump(vocab_relation, f, indent=4)
            print('Write into %s' % args.vocab_object_json.format(args.dataset, args.dataset))
            with open(args.vocab_object_json.format(args.dataset, args.dataset), 'w') as f:
                json.dump(vocab_object, f, indent=4)
        else:
            print('Loading vocab')
            with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
                vocab = json.load(f)
            print('Loading oie_subject vocab')
            with open(args.vocab_subject_json.format(args.dataset, args.dataset), 'r') as f:
                vocab_subject = json.load(f)
            print('Loading oie_relation vocab')
            with open(args.vocab_relation_json.format(args.dataset, args.dataset), 'r') as f:
                vocab_relation = json.load(f)
            print('Loading oie_object vocab')
            with open(args.vocab_object_json.format(args.dataset, args.dataset), 'r') as f:
                vocab_object = json.load(f)


            # Encode all questions
        print('Encoding data')
        questions_encoded_subject = []
        questions_encoded_relation = [] 
        questions_encoded_object = [] 
        questions_len_subject = []
        questions_len_relation = []
        questions_len_object = []
        questions_encoded = []
        questions_len = []
        question_ids = []
        video_ids_tbw = []
        video_names_tbw = []
        all_answers = []
        for idx, instance in enumerate(instances):
            question = instance['question'].lower()[:-1]
                #question = instance['q_body'].lower()[:-1]
            if client.annotate(question)==[]:
                q=question.split()
                token_subject = ' '.join(q[0:round(len(q)/3)])
                token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                token_object = ' '.join(q[2*round(len(q)/3):len(q)])
            else:
                token_subject = client.annotate(question)[0]['subject']
                token_relation = client.annotate(question)[0]['relation']
                token_object = client.annotate(question)[0]['object']

            if token_subject==[]:
                token_subject=question
            if token_relation==[]:
                token_relation=question
            if token_object ==[]:
                token_object=question

            subject_tokens = nltk.word_tokenize(token_subject)
            relation_tokens = nltk.word_tokenize(token_relation)
            object_tokens = nltk.word_tokenize(token_object)
            question_subject_encoded = utils.encode(subject_tokens, vocab_subject['question_token_to_idx'], allow_unk=True)
            question_relation_encoded = utils.encode(relation_tokens, vocab_relation['question_token_to_idx'], allow_unk=True)
            question_object_encoded = utils.encode(object_tokens, vocab_object['question_token_to_idx'], allow_unk=True)

            questions_encoded_subject.append(question_subject_encoded)
            questions_len_subject.append(len(question_subject_encoded))
            questions_encoded_relation.append(question_relation_encoded)
            questions_len_relation.append(len(question_relation_encoded))
            questions_encoded_object.append(question_object_encoded)
            questions_len_object.append(len(question_object_encoded))

            question_tokens = nltk.word_tokenize(question)
            question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)

            questions_encoded.append(question_encoded)
            questions_len.append(len(question_encoded))
            question_ids.append(idx)

            im_name=instance['video_id']
                #im_name = instance['vid_id']
            video_ids_tbw.append(im_name)
            video_names_tbw.append(im_name)
            # options: list = data[5:9]
            # answer_idx: int = data[9]
            # answer: str = options[answer_idx]
            if instance['answer'] in vocab['answer_token_to_idx']:
                answer = vocab['answer_token_to_idx'][instance['answer']]
            elif args.mode in ['train']:
                answer = 0
            elif args.mode in ['val', 'test']:
                answer = 1

            all_answers.append(answer)
        max_question_subject_length = max(len(x) for x in questions_encoded_subject)
        for qe in questions_encoded_subject:
            while len(qe) < max_question_subject_length:
                qe.append(vocab_subject['question_token_to_idx']['<NULL>'])

        max_question_relation_length = max(len(x) for x in questions_encoded_relation)
        for qe in questions_encoded_relation:
            while len(qe) < max_question_relation_length:
                qe.append(vocab_relation['question_token_to_idx']['<NULL>'])

        max_question_object_length = max(len(x) for x in questions_encoded_object)
        for qe in questions_encoded_object:
            while len(qe) < max_question_object_length:
                qe.append(vocab_object['question_token_to_idx']['<NULL>'])

        max_question_length = max(len(x) for x in questions_encoded)
        for qe in questions_encoded:
            while len(qe) < max_question_length:
                qe.append(vocab['question_token_to_idx']['<NULL>'])

        questions_encoded_subject = np.asarray(questions_encoded_subject, dtype=np.int32)
        questions_len_subject = np.asarray(questions_len_subject, dtype=np.int32)
        print(questions_encoded_subject.shape)

        questions_encoded_relation = np.asarray(questions_encoded_relation, dtype=np.int32)
        questions_len_relation = np.asarray(questions_len_relation, dtype=np.int32)
        print(questions_encoded_relation.shape)

        questions_encoded_object = np.asarray(questions_encoded_object, dtype=np.int32)
        questions_len_object = np.asarray(questions_len_object, dtype=np.int32)
        print(questions_encoded_object.shape)

        questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
        questions_len = np.asarray(questions_len, dtype=np.int32)
        print(questions_encoded.shape)

        glove_matrix = None
        glove_matrix_subject = None
        glove_matrix_relation = None
        glove_matrix_object = None
        if args.mode == 'train':
            token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
            token_itow_subject = {i: w for w, i in vocab_subject['question_token_to_idx'].items()}
            token_itow_relation = {i: w for w, i in vocab_relation['question_token_to_idx'].items()}
            token_itow_object = {i: w for w, i in vocab_object['question_token_to_idx'].items()}
            print("Load glove from %s" % args.glove_pt)
            glove = pickle.load(open(args.glove_pt, 'rb'))
            dim_word = glove['the'].shape[0]
            glove_matrix = []
            glove_matrix_subject = []
            glove_matrix_relation = []
            glove_matrix_object = []
            for i in range(len(token_itow)):
                vector = glove.get(token_itow[i], np.zeros((dim_word,)))
                glove_matrix.append(vector)
            glove_matrix = np.asarray(glove_matrix, dtype=np.float32)

            for i in range(len(token_itow_subject)):
                vector = glove.get(token_itow_subject[i], np.zeros((dim_word,)))
                glove_matrix_subject.append(vector)
            glove_matrix_subject = np.asarray(glove_matrix_subject, dtype=np.float32)

            for i in range(len(token_itow_relation)):
                vector = glove.get(token_itow_relation[i], np.zeros((dim_word,)))
                glove_matrix_relation.append(vector)
            glove_matrix_relation = np.asarray(glove_matrix_relation, dtype=np.float32)

            for i in range(len(token_itow_object)):
                vector = glove.get(token_itow_object[i], np.zeros((dim_word,)))
                glove_matrix_object.append(vector)
            glove_matrix_object = np.asarray(glove_matrix_object, dtype=np.float32)
            print(glove_matrix_object.shape)

        print('Writing', args.output_pt.format(args.dataset, args.dataset, args.mode))
        obj = {
                'questions': questions_encoded,
                'questions_len': questions_len,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix,
            }
        with open(args.output_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_subject_pt.format(args.dataset, args.dataset, args.mode))
        obj = {
                'questions': questions_encoded_subject,
                'questions_len': questions_len_subject,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_subject,
            }
        with open(args.output_subject_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_relation_pt.format(args.dataset, args.dataset, args.mode))
        obj = {
                'questions': questions_encoded_relation,
                'questions_len': questions_len_relation,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_relation,
            }
        with open(args.output_relation_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
            pickle.dump(obj, f)

        print('Writing', args.output_object_pt.format(args.dataset, args.dataset, args.mode))
        obj = {
                'questions': questions_encoded_object,
                'questions_len': questions_len_object,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_object,
            }
        with open(args.output_object_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
            pickle.dump(obj, f)

def process_questions_oie_keybert2(args):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        instances = json.load(dataset_file)
    kw_model = KeyBERT()  
    #roberta = TransformerDocumentEmbeddings('roberta-base')
    #kw_model = KeyBERT(model=roberta)
    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        print('Building vocab')
        answer_cnt = {}
        for instance in instances:
            answer = instance['answer']
            answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

        answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        answer_counter = Counter(answer_cnt)
        frequent_answers = answer_counter.most_common(args.answer_top)
        total_ans = sum(item[1] for item in answer_counter.items())
        total_freq_ans = sum(item[1] for item in frequent_answers)
        print("Number of unique answers:", len(answer_counter))
        print("Total number of answers:", total_ans)
        print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans * 100.0 / total_ans))

        for token, cnt in Counter(answer_cnt).most_common(args.answer_top):
            answer_token_to_idx[token] = len(answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))
        
        
        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        question_oie_subject_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        question_oie_relation_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        question_oie_object_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, instance in enumerate(instances):
            question = instance['question'].lower()[:-1]
            keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
            
            if keywords==[]:
                q=question.split()
                token_subject = ' '.join(q[0:round(len(q)/3)])
                token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                token_object = ' '.join(q[2*round(len(q)/3):len(q)])
            else:
                token_subject = ''.join(keywords[0][0])
                token_relation =''.join(keywords[1][0])
                token_object = ''.join(keywords[2][0])

            for token in nltk.word_tokenize(token_subject):
                if token not in question_oie_subject_token_to_idx:
                    question_oie_subject_token_to_idx[token] = len(question_oie_subject_token_to_idx)
            for token in nltk.word_tokenize(token_relation):
                if token not in question_oie_relation_token_to_idx:
                    question_oie_relation_token_to_idx[token] = len(question_oie_relation_token_to_idx)
            for token in nltk.word_tokenize(token_object):
                if token not in question_oie_object_token_to_idx:
                    question_oie_object_token_to_idx[token] = len(question_oie_object_token_to_idx)
                    #question = instance['q_body'].lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))
        print('Get question_oie_subject_token_to_idx')
        print(len(question_oie_subject_token_to_idx))
        print('Get question_oie_relation_token_to_idx')
        print(len(question_oie_relation_token_to_idx))
        print('Get question_oie_object_token_to_idx')
        print(len(question_oie_object_token_to_idx))

        vocab = {
                    'question_token_to_idx': question_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

        vocab_subject = {
                    'question_token_to_idx': question_oie_subject_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

        vocab_relation = {
                    'question_token_to_idx': question_oie_relation_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

        vocab_object = {
                    'question_token_to_idx': question_oie_object_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

        print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
        with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab, f, indent=4)
        print('Write into %s' % args.vocab_subject_json.format(args.dataset, args.dataset))
        with open(args.vocab_subject_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab_subject, f, indent=4)
        print('Write into %s' % args.vocab_relation_json.format(args.dataset, args.dataset))
        with open(args.vocab_relation_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab_relation, f, indent=4)
        print('Write into %s' % args.vocab_object_json.format(args.dataset, args.dataset))
        with open(args.vocab_object_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab_object, f, indent=4)            

    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
            vocab = json.load(f)
        print('Loading oie_subject vocab')
        with open(args.vocab_subject_json.format(args.dataset, args.dataset), 'r') as f:
            vocab_subject = json.load(f)
        print('Loading oie_relation vocab')
        with open(args.vocab_relation_json.format(args.dataset, args.dataset), 'r') as f:
            vocab_relation = json.load(f)
        print('Loading oie_object vocab')
        with open(args.vocab_object_json.format(args.dataset, args.dataset), 'r') as f:
            vocab_object = json.load(f)


    # Encode all questions
    print('Encoding data')
    questions_encoded_subject = []
    questions_encoded_relation = [] 
    questions_encoded_object = [] 
    subject_input_batch_bert= []
    subject_attention_mask_batch_bert= []
    subject_token_type_ids_batch_bert= []
    relation_input_batch_bert= []
    relation_attention_mask_batch_bert= []
    relation_token_type_ids_batch_bert= []
    object_input_batch_bert= []
    object_attention_mask_batch_bert= []
    object_token_type_ids_batch_bert= []
    question_input_batch_bert= []
    question_attention_mask_batch_bert= []
    question_token_type_ids_batch_bert= []
    questions_len_subject = []
    questions_len_relation = []
    questions_len_object = []
    subject_bert_len = []
    relation_bert_len = []
    object_bert_len = []
    questions_encoded = []
    questions_len = []
    question_bert_len = []
    question_ids = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    answers_encoded = []
    answers_input_batch_bert= []
    answers_attention_mask_batch_bert= []
    answers_token_type_ids_batch_bert= []
    answers_bert_len = []
    
    for idx, instance in enumerate(instances):
        question = instance['question'].lower()[:-1]
                #question = instance['q_body'].lower()[:-1]
        keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
        if keywords==[]:
            q=question.split()
            token_subject = ' '.join(q[0:round(len(q)/3)])
            token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
            token_object = ' '.join(q[2*round(len(q)/3):len(q)])
        else:
            token_subject = ''.join(keywords[0][0])
            token_relation =''.join(keywords[1][0])
            token_object = ''.join(keywords[2][0])

        subject_tokens = nltk.word_tokenize(token_subject)
        relation_tokens = nltk.word_tokenize(token_relation)
        object_tokens = nltk.word_tokenize(token_object)
        question_subject_encoded = utils.encode(subject_tokens, vocab_subject['question_token_to_idx'], allow_unk=True)
        question_relation_encoded = utils.encode(relation_tokens, vocab_relation['question_token_to_idx'], allow_unk=True)
        question_object_encoded = utils.encode(object_tokens, vocab_object['question_token_to_idx'], allow_unk=True)

        subject_tokens_dict = tokenizer([token_subject], padding=True)
        subject_input_batch = subject_tokens_dict["input_ids"]
        subject_attention_mask_batch = subject_tokens_dict["attention_mask"]
        subject_token_type_ids_batch = subject_tokens_dict["token_type_ids"]
            # subject_input_var = torch.LongTensor(subject_input_batch)
            # subject_attention_mask_var = torch.LongTensor(subject_attention_mask_batch)
            # subject_token_type_ids_var = torch.LongTensor(subject_token_type_ids_batch)
            # pooled_output  = model(input_ids=subject_input_var, attention_mask=subject_attention_mask_var,token_type_ids=subject_token_type_ids_var)  
            
        relation_tokens_dict = tokenizer([token_relation], padding=True)
        relation_input_batch = relation_tokens_dict["input_ids"]
        relation_attention_mask_batch = relation_tokens_dict["attention_mask"]
        relation_token_type_ids_batch = relation_tokens_dict["token_type_ids"]
        object_tokens_dict = tokenizer([token_object], padding=True)
        object_input_batch = object_tokens_dict["input_ids"]
        object_attention_mask_batch = object_tokens_dict["attention_mask"]
        object_token_type_ids_batch = object_tokens_dict["token_type_ids"]

        questions_encoded_subject.append(question_subject_encoded)
        questions_len_subject.append(len(question_subject_encoded))
        questions_encoded_relation.append(question_relation_encoded)
        questions_len_relation.append(len(question_relation_encoded))
        questions_encoded_object.append(question_object_encoded)
        questions_len_object.append(len(question_object_encoded))

        subject_input_batch_bert.append(subject_input_batch)
        subject_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(subject_input_batch))).squeeze().size())
        subject_attention_mask_batch_bert.append(subject_attention_mask_batch)
        subject_token_type_ids_batch_bert.append(subject_token_type_ids_batch)
        relation_input_batch_bert.append(relation_input_batch)
        relation_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(relation_input_batch))).squeeze().size())
        relation_attention_mask_batch_bert.append(relation_attention_mask_batch)
        relation_token_type_ids_batch_bert.append(relation_token_type_ids_batch)
        object_input_batch_bert.append(object_input_batch)
        object_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(object_input_batch))).squeeze().size())
        object_attention_mask_batch_bert.append(object_attention_mask_batch)
        object_token_type_ids_batch_bert.append(object_token_type_ids_batch)

        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
        question_tokens_dict = tokenizer([question], padding=True)
        question_input_batch = question_tokens_dict["input_ids"]
        question_attention_mask_batch = question_tokens_dict["attention_mask"]
        question_token_type_ids_batch = question_tokens_dict["token_type_ids"]

        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_input_batch_bert.append(question_input_batch)
        question_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(question_input_batch))).squeeze().size())
        question_attention_mask_batch_bert.append(question_attention_mask_batch)
        question_token_type_ids_batch_bert.append(question_token_type_ids_batch)
        question_ids.append(idx)

        im_name=instance['video_id']
                #im_name = instance['vid_id']
        video_ids_tbw.append(im_name)
        video_names_tbw.append(im_name)
            # options: list = data[5:9]
            # answer_idx: int = data[9]
            # answer: str = options[answer_idx]
        if instance['answer'] in vocab['answer_token_to_idx']:
            answer = vocab['answer_token_to_idx'][instance['answer']]
        elif args.mode in ['train']:
            answer = 0
        elif args.mode in ['val', 'test']:
            answer = 1

        all_answers.append(answer)

    for answer_token, answer_idx in vocab_subject['answer_token_to_idx'].items():
        # Tokenize and encode the answer token using the BERT tokenizer
        input_text=answer_token.lower()
        
        inputs = tokenizer([input_text], padding=True)
        input_ids = np.squeeze(inputs['input_ids']).tolist()
        
        attention_mask = np.squeeze(inputs['attention_mask']).tolist()
        
        token_type_ids = np.squeeze(inputs['token_type_ids']).tolist()
        
    
        # Store the encoded answer token and its associated input data
        #answers_encoded.append(inputs)
        answers_input_batch_bert.append(input_ids)
        answers_attention_mask_batch_bert.append(attention_mask)
        answers_token_type_ids_batch_bert.append(token_type_ids)
        answers_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(input_ids))).squeeze().size())
    print(max(answers_bert_len))
    max_question_length = 8
    for qe in answers_input_batch_bert:    
        while len(qe) < max_question_length:
            qe.append(vocab['answer_token_to_idx']['<NULL>'])
        
    
    max_question_length = 8
    for qe in answers_attention_mask_batch_bert:
        while len(qe) < max_question_length:
            qe.append(vocab['answer_token_to_idx']['<NULL>'])
    
    max_question_length = 8
    for qe in answers_token_type_ids_batch_bert:
        while len(qe) < max_question_length:
            qe.append(vocab['answer_token_to_idx']['<NULL>'])
    
    
    
    
    
    
    max_question_subject_length = max(len(x) for x in questions_encoded_subject)
    for qe in questions_encoded_subject:
        while len(qe) < max_question_subject_length:
            qe.append(vocab_subject['question_token_to_idx']['<NULL>'])

    max_question_relation_length = max(len(x) for x in questions_encoded_relation)
    for qe in questions_encoded_relation:
        while len(qe) < max_question_relation_length:
            qe.append(vocab_relation['question_token_to_idx']['<NULL>'])

    max_question_object_length = max(len(x) for x in questions_encoded_object)
    for qe in questions_encoded_object:
        while len(qe) < max_question_object_length:
            qe.append(vocab_object['question_token_to_idx']['<NULL>'])

    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    max_subject_input_batch_bert_length = max(len(x) for x in subject_input_batch_bert)
    for qe in subject_input_batch_bert:
        while len(qe) < max_subject_input_batch_bert_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    max_subject_attention_mask_batch_bert_length = max(len(x) for x in subject_attention_mask_batch_bert)
    for qe in subject_attention_mask_batch_bert:
        while len(qe) < max_subject_attention_mask_batch_bert_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    max_subject_token_type_ids_batch_bert_length = max(len(x) for x in subject_token_type_ids_batch_bert)
    for qe in subject_token_type_ids_batch_bert:
        while len(qe) < max_subject_token_type_ids_batch_bert_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    max_relation_input_batch_bert_length = max(len(x) for x in relation_input_batch_bert)
    for qe in relation_input_batch_bert:
        while len(qe) < max_relation_input_batch_bert_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    max_relation_attention_mask_batch_bert_length = max(len(x) for x in relation_attention_mask_batch_bert)
    for qe in relation_attention_mask_batch_bert:
        while len(qe) < max_relation_attention_mask_batch_bert_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    max_relation_token_type_ids_batch_bert_length = max(len(x) for x in relation_token_type_ids_batch_bert)
    for qe in relation_token_type_ids_batch_bert:
        while len(qe) < max_relation_token_type_ids_batch_bert_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    max_object_input_batch_bert_length = max(len(x) for x in object_input_batch_bert)
    for qe in object_input_batch_bert:
        while len(qe) < max_object_input_batch_bert_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    max_object_attention_mask_batch_bert_length = max(len(x) for x in object_attention_mask_batch_bert)
    for qe in object_attention_mask_batch_bert:
        while len(qe) < max_object_attention_mask_batch_bert_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    max_object_token_type_ids_batch_bert_length = max(len(x) for x in object_token_type_ids_batch_bert)
    for qe in object_token_type_ids_batch_bert:
        while len(qe) < max_object_token_type_ids_batch_bert_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])    

    questions_encoded_subject = np.asarray(questions_encoded_subject, dtype=np.int32)
    questions_len_subject = np.asarray(questions_len_subject, dtype=np.int32)
    print(questions_encoded_subject.shape)

    questions_encoded_relation = np.asarray(questions_encoded_relation, dtype=np.int32)
    questions_len_relation = np.asarray(questions_len_relation, dtype=np.int32)
    print(questions_encoded_relation.shape)

    questions_encoded_object = np.asarray(questions_encoded_object, dtype=np.int32)
    questions_len_object = np.asarray(questions_len_object, dtype=np.int32)
    print(questions_encoded_object.shape)

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    glove_matrix = None
    glove_matrix_subject = None
    glove_matrix_relation = None
    glove_matrix_object = None
    if args.mode == 'train':
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        token_itow_subject = {i: w for w, i in vocab_subject['question_token_to_idx'].items()}
        token_itow_relation = {i: w for w, i in vocab_relation['question_token_to_idx'].items()}
        token_itow_object = {i: w for w, i in vocab_object['question_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        glove_matrix_subject = []
        glove_matrix_relation = []
        glove_matrix_object = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)

        for i in range(len(token_itow_subject)):
            vector = glove.get(token_itow_subject[i], np.zeros((dim_word,)))
            glove_matrix_subject.append(vector)
        glove_matrix_subject = np.asarray(glove_matrix_subject, dtype=np.float32)

        for i in range(len(token_itow_relation)):
            vector = glove.get(token_itow_relation[i], np.zeros((dim_word,)))
            glove_matrix_relation.append(vector)
        glove_matrix_relation = np.asarray(glove_matrix_relation, dtype=np.float32)

        for i in range(len(token_itow_object)):
            vector = glove.get(token_itow_object[i], np.zeros((dim_word,)))
            glove_matrix_object.append(vector)
        glove_matrix_object = np.asarray(glove_matrix_object, dtype=np.float32)
        print(glove_matrix_object.shape)

    print('Writing', args.output_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
                'questions': questions_encoded,
                'question_input_bert': question_input_batch_bert,   
                'question_mask_bert': question_attention_mask_batch_bert,
                'question_ids_bert': question_token_type_ids_batch_bert,        
                'question_bert_len': question_bert_len,  
                'questions_len': questions_len,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'answers_input_bert': answers_input_batch_bert,
                'answers_mask_bert': answers_attention_mask_batch_bert,
                'answers_ids_bert': answers_token_type_ids_batch_bert,
                'answers_bert_len': answers_bert_len,
                'glove': glove_matrix,
            }
    with open(args.output_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)

    print('Writing', args.output_subject_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
                'questions': questions_encoded_subject,
                'question_input_bert': question_input_batch_bert,   
                'question_mask_bert': question_attention_mask_batch_bert,
                'question_ids_bert': question_token_type_ids_batch_bert,        
                'question_bert_len': question_bert_len,  
                'questions_len': questions_len_subject,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_subject,
            }
    with open(args.output_subject_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)

    print('Writing', args.output_relation_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
                'questions': questions_encoded_relation,
                'question_input_bert': question_input_batch_bert,   
                'question_mask_bert': question_attention_mask_batch_bert,
                'question_ids_bert': question_token_type_ids_batch_bert,        
                'question_bert_len': question_bert_len,  
                'questions_len': questions_len_relation,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_relation,
            }
    with open(args.output_relation_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)

    print('Writing', args.output_object_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
                'questions': questions_encoded_object,
                'question_input_bert': question_input_batch_bert,   
                'question_mask_bert': question_attention_mask_batch_bert,
                'question_ids_bert': question_token_type_ids_batch_bert,        
                'question_bert_len': question_bert_len,  
                'questions_len': questions_len_object,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_object,
            }
    with open(args.output_object_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)

def process_questions_oie_keybert(args):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        instances = json.load(dataset_file)
    kw_model = KeyBERT()  
    #roberta = TransformerDocumentEmbeddings('roberta-base')
    #kw_model = KeyBERT(model=roberta)
    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        print('Building vocab')
        answer_cnt = {}
        for instance in instances:
            answer = instance['answer']
            answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

        answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
        answer_counter = Counter(answer_cnt)
        frequent_answers = answer_counter.most_common(args.answer_top)
        total_ans = sum(item[1] for item in answer_counter.items())
        total_freq_ans = sum(item[1] for item in frequent_answers)
        print("Number of unique answers:", len(answer_counter))
        print("Total number of answers:", total_ans)
        print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans * 100.0 / total_ans))

        for token, cnt in Counter(answer_cnt).most_common(args.answer_top):
            answer_token_to_idx[token] = len(answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        question_oie_subject_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        question_oie_relation_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        question_oie_object_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, instance in enumerate(instances):
            question = instance['question'].lower()[:-1]
            keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
            print(keywords)
            if keywords==[]:
                q=question.split()
                token_subject = ' '.join(q[0:round(len(q)/3)])
                token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
                token_object = ' '.join(q[2*round(len(q)/3):len(q)])
            else:
                token_subject = ''.join(keywords[0][0])
                token_relation =''.join(keywords[1][0])
                token_object = ''.join(keywords[2][0])

            for token in nltk.word_tokenize(token_subject):
                if token not in question_oie_subject_token_to_idx:
                    question_oie_subject_token_to_idx[token] = len(question_oie_subject_token_to_idx)
            for token in nltk.word_tokenize(token_relation):
                if token not in question_oie_relation_token_to_idx:
                    question_oie_relation_token_to_idx[token] = len(question_oie_relation_token_to_idx)
            for token in nltk.word_tokenize(token_object):
                if token not in question_oie_object_token_to_idx:
                    question_oie_object_token_to_idx[token] = len(question_oie_object_token_to_idx)
                    #question = instance['q_body'].lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))
        print('Get question_oie_subject_token_to_idx')
        print(len(question_oie_subject_token_to_idx))
        print('Get question_oie_relation_token_to_idx')
        print(len(question_oie_relation_token_to_idx))
        print('Get question_oie_object_token_to_idx')
        print(len(question_oie_object_token_to_idx))

        vocab = {
                    'question_token_to_idx': question_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

        vocab_subject = {
                    'question_token_to_idx': question_oie_subject_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

        vocab_relation = {
                    'question_token_to_idx': question_oie_relation_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

        vocab_object = {
                    'question_token_to_idx': question_oie_object_token_to_idx,
                    'answer_token_to_idx': answer_token_to_idx,
                    'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
                }

        print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
        with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab, f, indent=4)
        print('Write into %s' % args.vocab_subject_json.format(args.dataset, args.dataset))
        with open(args.vocab_subject_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab_subject, f, indent=4)
        print('Write into %s' % args.vocab_relation_json.format(args.dataset, args.dataset))
        with open(args.vocab_relation_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab_relation, f, indent=4)
        print('Write into %s' % args.vocab_object_json.format(args.dataset, args.dataset))
        with open(args.vocab_object_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab_object, f, indent=4)            

    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
            vocab = json.load(f)
        print('Loading oie_subject vocab')
        with open(args.vocab_subject_json.format(args.dataset, args.dataset), 'r') as f:
            vocab_subject = json.load(f)
        print('Loading oie_relation vocab')
        with open(args.vocab_relation_json.format(args.dataset, args.dataset), 'r') as f:
            vocab_relation = json.load(f)
        print('Loading oie_object vocab')
        with open(args.vocab_object_json.format(args.dataset, args.dataset), 'r') as f:
            vocab_object = json.load(f)


            # Encode all questions
    print('Encoding data')
    questions_encoded_subject = []
    questions_encoded_relation = [] 
    questions_encoded_object = [] 
    questions_len_subject = []
    questions_len_relation = []
    questions_len_object = []
    questions_encoded = []
    questions_len = []
    question_ids = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    for idx, instance in enumerate(instances):
        question = instance['question'].lower()[:-1]
                #question = instance['q_body'].lower()[:-1]
        keywords = kw_model.extract_keywords(question, keyphrase_ngram_range=(1, 3), stop_words=None, use_mmr=True, diversity=1,top_n=3)
        if keywords==[]:
            q=question.split()
            token_subject = ' '.join(q[0:round(len(q)/3)])
            token_relation = ' '.join(q[round(len(q)/3):2*round(len(q)/3)])
            token_object = ' '.join(q[2*round(len(q)/3):len(q)])
        else:
            token_subject = ''.join(keywords[0][0])
            token_relation =''.join(keywords[1][0])
            token_object = ''.join(keywords[2][0])

        subject_tokens = nltk.word_tokenize(token_subject)
        relation_tokens = nltk.word_tokenize(token_relation)
        object_tokens = nltk.word_tokenize(token_object)
        question_subject_encoded = utils.encode(subject_tokens, vocab_subject['question_token_to_idx'], allow_unk=True)
        question_relation_encoded = utils.encode(relation_tokens, vocab_relation['question_token_to_idx'], allow_unk=True)
        question_object_encoded = utils.encode(object_tokens, vocab_object['question_token_to_idx'], allow_unk=True)

        questions_encoded_subject.append(question_subject_encoded)
        questions_len_subject.append(len(question_subject_encoded))
        questions_encoded_relation.append(question_relation_encoded)
        questions_len_relation.append(len(question_relation_encoded))
        questions_encoded_object.append(question_object_encoded)
        questions_len_object.append(len(question_object_encoded))

        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)

        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)

        im_name=instance['video_id']
                #im_name = instance['vid_id']
        video_ids_tbw.append(im_name)
        video_names_tbw.append(im_name)
            # options: list = data[5:9]
            # answer_idx: int = data[9]
            # answer: str = options[answer_idx]
        if instance['answer'] in vocab['answer_token_to_idx']:
            answer = vocab['answer_token_to_idx'][instance['answer']]
        elif args.mode in ['train']:
            answer = 0
        elif args.mode in ['val', 'test']:
            answer = 1

        all_answers.append(answer)
    max_question_subject_length = max(len(x) for x in questions_encoded_subject)
    for qe in questions_encoded_subject:
        while len(qe) < max_question_subject_length:
            qe.append(vocab_subject['question_token_to_idx']['<NULL>'])

    max_question_relation_length = max(len(x) for x in questions_encoded_relation)
    for qe in questions_encoded_relation:
        while len(qe) < max_question_relation_length:
            qe.append(vocab_relation['question_token_to_idx']['<NULL>'])

    max_question_object_length = max(len(x) for x in questions_encoded_object)
    for qe in questions_encoded_object:
        while len(qe) < max_question_object_length:
            qe.append(vocab_object['question_token_to_idx']['<NULL>'])

    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    questions_encoded_subject = np.asarray(questions_encoded_subject, dtype=np.int32)
    questions_len_subject = np.asarray(questions_len_subject, dtype=np.int32)
    print(questions_encoded_subject.shape)

    questions_encoded_relation = np.asarray(questions_encoded_relation, dtype=np.int32)
    questions_len_relation = np.asarray(questions_len_relation, dtype=np.int32)
    print(questions_encoded_relation.shape)

    questions_encoded_object = np.asarray(questions_encoded_object, dtype=np.int32)
    questions_len_object = np.asarray(questions_len_object, dtype=np.int32)
    print(questions_encoded_object.shape)

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    glove_matrix = None
    glove_matrix_subject = None
    glove_matrix_relation = None
    glove_matrix_object = None
    if args.mode == 'train':
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        token_itow_subject = {i: w for w, i in vocab_subject['question_token_to_idx'].items()}
        token_itow_relation = {i: w for w, i in vocab_relation['question_token_to_idx'].items()}
        token_itow_object = {i: w for w, i in vocab_object['question_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        glove_matrix_subject = []
        glove_matrix_relation = []
        glove_matrix_object = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)

        for i in range(len(token_itow_subject)):
            vector = glove.get(token_itow_subject[i], np.zeros((dim_word,)))
            glove_matrix_subject.append(vector)
        glove_matrix_subject = np.asarray(glove_matrix_subject, dtype=np.float32)

        for i in range(len(token_itow_relation)):
            vector = glove.get(token_itow_relation[i], np.zeros((dim_word,)))
            glove_matrix_relation.append(vector)
        glove_matrix_relation = np.asarray(glove_matrix_relation, dtype=np.float32)

        for i in range(len(token_itow_object)):
            vector = glove.get(token_itow_object[i], np.zeros((dim_word,)))
            glove_matrix_object.append(vector)
        glove_matrix_object = np.asarray(glove_matrix_object, dtype=np.float32)
        print(glove_matrix_object.shape)

    print('Writing', args.output_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
                'questions': questions_encoded,
                'questions_len': questions_len,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix,
            }
    with open(args.output_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)

    print('Writing', args.output_subject_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
                'questions': questions_encoded_subject,
                'questions_len': questions_len_subject,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_subject,
            }
    with open(args.output_subject_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)

    print('Writing', args.output_relation_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
                'questions': questions_encoded_relation,
                'questions_len': questions_len_relation,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_relation,
            }
    with open(args.output_relation_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)

    print('Writing', args.output_object_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
                'questions': questions_encoded_object,
                'questions_len': questions_len_object,
                'question_id': question_ids,
                'video_ids': np.asarray(video_ids_tbw),
                'video_names': np.array(video_names_tbw),
                'answers': all_answers,
                'glove': glove_matrix_object,
            }
    with open(args.output_object_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)

def load_file(filename):
    """
    load obj from filename
    :param filename:
    :return:
    """
    cont = None
    if not osp.exists(filename):
        print('{} not exist'.format(filename))
        return cont
    if osp.splitext(filename)[-1] == '.csv':
        # return pd.read_csv(filename, delimiter= '\t', index_col=0)
        return pd.read_csv(filename, delimiter=',')
    with open(filename, 'r') as fp:
        if osp.splitext(filename)[1] == '.txt':
            cont = fp.readlines()
            cont = [c.rstrip('\n') for c in cont]
        elif osp.splitext(filename)[1] == '.json':
            cont = json.load(fp)
    return cont

def process_questions_oie_keybert_bert3(args):
    ''' Encode question tokens'''
    print('Loading data')
    
    with open(args.annotation_file) as dataset_file:
        instances = dataset_file.readlines()
    _header = instances.pop(0)
   
    #roberta = TransformerDocumentEmbeddings('roberta-base')
    #kw_model = KeyBERT(model=roberta)
    # Either create the vocab or load it from disk
    # properties = {
    #     'openie.affinity_probability_cap': 2/3,
    #                     }
    # with StanfordOpenIE(properties=properties) as client:
    if args.mode in ['train']:
        print('Building vocab')
        answer_cnt = {}

        for instance in instances:
            
            data: list = list(instance.strip().split(",")) 
            options: list = data[8:13]#5个选项的文本
            answer_idx: int = int(data[5])#正确答案选项
            answer: str = options[answer_idx]#正确答案文本
            #answer = instance['answer']
            answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

        answer_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        answer_counter = Counter(answer_cnt)
        frequent_answers = answer_counter.most_common(args.answer_top)
        total_ans = sum(item[1] for item in answer_counter.items())
        total_freq_ans = sum(item[1] for item in frequent_answers)
        print("Number of unique answers:", len(answer_counter))
        print("Total number of answers:", total_ans)
        print("Top %i answers account for %f%%" % (len(frequent_answers), total_freq_ans * 100.0 / total_ans))

        question_answer_token_to_idx={'<NULL>': 0, '<UNK>': 1}
        for i, instance in enumerate(instances):
            data: list = list(instance.strip().split(",")) 
            answer1: str = data[8].lower()
            answer2: str = data[9].lower()
            answer3: str = data[10].lower()
            answer4: str = data[11].lower()
            answer5: str = data[12].lower()
            question: str = data[4].lower()
            ques_type: str = data[7].lower()


            for token in nltk.word_tokenize(answer1):
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
            for token in nltk.word_tokenize(answer2):
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
            for token in nltk.word_tokenize(answer3):
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
            for token in nltk.word_tokenize(answer4):
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
            for token in nltk.word_tokenize(answer5):
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
            for token in nltk.word_tokenize(question):
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)
            for token in nltk.word_tokenize(ques_type):
                if token not in question_answer_token_to_idx:
                    question_answer_token_to_idx[token] = len(question_answer_token_to_idx)


        print('Get question_answer_token_to_idx')
        print(len(question_answer_token_to_idx))


        vocab = {
                'question_token_to_idx': question_answer_token_to_idx,
                'answer_token_to_idx': question_answer_token_to_idx,
                'question_answer_token_to_idx': question_answer_token_to_idx
            }


        print('Write into %s' % args.vocab_json.format(args.dataset, args.dataset))
        with open(args.vocab_json.format(args.dataset, args.dataset), 'w') as f:
            json.dump(vocab, f, indent=4)

    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset, args.dataset), 'r') as f:
            vocab = json.load(f)


    # Encode all questions
    print('Encoding data')


    question_input_batch_bert= []
    question_attention_mask_batch_bert= []
    question_token_type_ids_batch_bert= []

    questions_type=[]
    questions_encoded = []
    questions_len = []
    question_bert_len = []
    question_ids = []

    answer_input_batch_bert= []
    answer_attention_mask_batch_bert= []
    answer_token_type_ids_batch_bert= []

    answer_encoded = []
    answer_len = []
    answer_bert_len = []
    answer_ids = []

    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    all_answer_cands_encoded = []
    all_answer_cands_len = []
    all_candidateds_bert_encoded = []
    all_candidates_bert_len = []
    all_candidates_attention_mask_batch_bert = []
    all_candidates_token_type_ids_batch_bert = []
    for idx, instance in enumerate(instances):
        data: list = list(instance.strip().split(",")) 
        question: str = data[4].lower()
        ques_type: str = data[7].lower()
        candidates1: str = data[8].lower()
        candidates2: str = data[9].lower()
        candidates3: str = data[10].lower()
        candidates4: str = data[11].lower()
        candidates5: str = data[12].lower()

        candidates=[]

        candidates.append(candidates1)
        candidates.append(candidates2)
        candidates.append(candidates3)
        candidates.append(candidates4)
        candidates.append(candidates5)


        question_type_tokens = nltk.word_tokenize(ques_type)

        question_type_encoded = utils.encode(question_type_tokens, vocab['question_token_to_idx'], allow_unk=True)

        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens, vocab['question_token_to_idx'], allow_unk=True)
        question_tokens_dict = tokenizer([question], padding=True)
        question_input_batch = question_tokens_dict["input_ids"]
        question_attention_mask_batch = question_tokens_dict["attention_mask"]
        question_token_type_ids_batch = question_tokens_dict["token_type_ids"]

        questions_type.append(question_type_encoded)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_input_batch_bert.append(question_input_batch)
        question_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(question_input_batch))).squeeze().size())
        question_attention_mask_batch_bert.append(question_attention_mask_batch)
        question_token_type_ids_batch_bert.append(question_token_type_ids_batch)
        question_ids.append(idx)


        answer_idx1: int = int(data[5])
        answer: str = candidates[answer_idx1]#正确答案文本

        answer_tokens1 = nltk.word_tokenize(answer)
        answer_encoded1 = utils.encode(answer_tokens1, vocab['answer_token_to_idx'], allow_unk=True)
        answer_tokens_dict1 = tokenizer([answer], padding=True)
        answer_input_batch1 = answer_tokens_dict1["input_ids"]
        answer_attention_mask_batch1 = answer_tokens_dict1["attention_mask"]
        answer_token_type_ids_batch1 = answer_tokens_dict1["token_type_ids"]

        answer_encoded.append(answer_encoded1)
        answer_len.append(len(answer_encoded1))
        answer_input_batch_bert.append(answer_input_batch1)
        answer_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(answer_input_batch1))).squeeze().size())
        answer_attention_mask_batch_bert.append(answer_attention_mask_batch1)
        answer_token_type_ids_batch_bert.append(answer_token_type_ids_batch1)
        answer_ids.append(idx)


        im_name=data[0]
            #im_name = instance['vid_id']
        video_ids_tbw.append(im_name)
        video_names_tbw.append(im_name)
        options: list = data[8:13]
        answer_idx: int = int(data[5])
        answer: str = options[answer_idx]
        if answer in vocab['answer_token_to_idx']:
            answer = vocab['answer_token_to_idx'][answer]
        elif args.mode in ['train']:
            answer = 0
        elif args.mode in ['val', 'test']:
            answer = 1

        all_answers.append(answer_idx)

        # answer candidates
        #candidates = ans_candidates[idx]
        candidates_encoded = []
        candidates_len = []
        candidateds_bert_encoded = []
        candidates_bert_len = []
        candidates_attention_mask_batch_bert = []
        candidates_token_type_ids_batch_bert = []
        for ans in candidates:
            ans = ans.lower()
            ans_tokens = nltk.word_tokenize(ans)
            cand_encoded = utils.encode(ans_tokens, vocab['question_answer_token_to_idx'], allow_unk=True)
            candidates_encoded.append(cand_encoded)
            candidates_len.append(len(cand_encoded))
            candidates_tokens_dict = tokenizer([ans], padding=True)
            candidates_input_batch = candidates_tokens_dict["input_ids"]
            candidates_attention_mask_batch = candidates_tokens_dict["attention_mask"]
            candidates_token_type_ids_batch = candidates_tokens_dict["token_type_ids"]
            candidateds_bert_encoded.append(candidates_input_batch)
            candidates_bert_len.append(torch.LongTensor(torch.from_numpy(np.asarray(candidates_input_batch))).squeeze().size())
            candidates_attention_mask_batch_bert.append(candidates_attention_mask_batch)
            candidates_token_type_ids_batch_bert.append(candidates_token_type_ids_batch)

        all_answer_cands_encoded.append(candidates_encoded)
        all_answer_cands_len.append(candidates_len)
        all_candidateds_bert_encoded.append(candidateds_bert_encoded)
        all_candidates_bert_len.append(candidates_bert_len)
        all_candidates_attention_mask_batch_bert.append(candidates_attention_mask_batch_bert)
        all_candidates_token_type_ids_batch_bert.append(candidates_token_type_ids_batch_bert)

    # Pad encoded answer candidates
    max_answer_cand_length1 = max(max(len(x) for x in candidate) for candidate in all_answer_cands_encoded)
    max_answer_cand_length2 = max(max(len(x[0]) for x in candidate) for candidate in all_candidateds_bert_encoded)
    max_answer_cand_length3 = max(max(len(x[0]) for x in candidate) for candidate in all_candidates_attention_mask_batch_bert)
    max_answer_cand_length4 = max(max(len(x[0]) for x in candidate) for candidate in all_candidates_token_type_ids_batch_bert)
    max_answer_length1 = max(len(x) for x in answer_encoded)
    max_answer_length2 = max(len(x) for x in answer_input_batch_bert)
    max_answer_length3 = max(len(x) for x in answer_attention_mask_batch_bert)
    max_answer_length4 = max(len(x) for x in answer_token_type_ids_batch_bert)


    max_question_length1 = max(len(x) for x in questions_encoded)
    max_question_length2 = max(len(x) for x in question_input_batch_bert)
    max_question_length3 = max(len(x) for x in question_attention_mask_batch_bert)
    max_question_length4 = max(len(x) for x in question_token_type_ids_batch_bert)
    max_value=max(max_answer_cand_length1,max_answer_cand_length2,max_answer_cand_length3,\
                  max_answer_cand_length4,max_answer_length1,max_question_length1,max_answer_length2,max_question_length2,\
                 max_answer_length3,max_question_length3,max_answer_length4,max_question_length4)
    print(max_value)
    max_value=32 #we find train 32, test 22, we select 32


    for ans_cands in all_answer_cands_encoded:
        for ans in ans_cands:
            while len(ans) < max_value:
                ans.append(vocab['question_answer_token_to_idx']['<NULL>'])

    for ans_cands in all_candidateds_bert_encoded:
        for ans in ans_cands:
            for ans_s in ans:
                while len(ans_s) < max_value:
                    ans_s.append(vocab['question_answer_token_to_idx']['<NULL>'])

    for ans_cands in all_candidates_attention_mask_batch_bert:
        for ans in ans_cands:
            for ans_s in ans:
                while len(ans_s) < max_value:
                    ans_s.append(vocab['question_answer_token_to_idx']['<NULL>'])

    for ans_cands in all_candidates_token_type_ids_batch_bert:
        for ans in ans_cands:
            for ans_s in ans:
                while len(ans_s) < max_value:
                    ans_s.append(vocab['question_answer_token_to_idx']['<NULL>'])
    all_answer_cands_encoded = np.asarray(all_answer_cands_encoded, dtype=np.int32)
    all_answer_cands_len = np.asarray(all_answer_cands_len, dtype=np.int32)
    all_candidateds_bert_encoded= all_candidateds_bert_encoded
    all_candidates_attention_mask_batch_bert=all_candidates_attention_mask_batch_bert
    all_candidates_token_type_ids_batch_bert=all_candidates_token_type_ids_batch_bert
    print(all_answer_cands_encoded.shape)


    for qe in questions_encoded:
        while len(qe) < max_value:
            qe.append(vocab['question_token_to_idx']['<NULL>'])
    for qe in question_input_batch_bert:
        for qe_s in qe:
            while len(qe_s) < max_value:
                qe_s.append(vocab['question_token_to_idx']['<NULL>'])
    for qe in question_attention_mask_batch_bert:
        for qe_s in qe:
            while len(qe_s) < max_value:
                qe_s.append(vocab['question_token_to_idx']['<NULL>'])
    for qe in question_token_type_ids_batch_bert:
        for qe_s in qe:
            while len(qe_s) < max_value:
                qe_s.append(vocab['question_token_to_idx']['<NULL>'])
    for qe in questions_type:

        while len(qe) < max_value:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    questions_type = np.asarray(questions_type, dtype=np.int32)
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    question_input_batch_bert=question_input_batch_bert
    question_attention_mask_batch_bert=question_attention_mask_batch_bert
    question_token_type_ids_batch_bert=question_token_type_ids_batch_bert
    print(questions_encoded.shape)


    question_bert_len = np.asarray(question_bert_len, dtype=np.int32)


    for qe in answer_encoded:
        while len(qe) < max_value:
            qe.append(vocab['answer_token_to_idx']['<NULL>'])
    for qe in answer_input_batch_bert:
        for qe_s in qe:
            while len(qe_s) < max_value:
                qe_s.append(vocab['question_token_to_idx']['<NULL>'])
    for qe in answer_attention_mask_batch_bert:
        for qe_s in qe:
            while len(qe_s) < max_value:
                qe_s.append(vocab['question_token_to_idx']['<NULL>'])
    for qe in answer_token_type_ids_batch_bert:
        for qe_s in qe:
            while len(qe_s) < max_value:
                qe_s.append(vocab['question_token_to_idx']['<NULL>'])

    answer_encoded = np.asarray(answer_encoded, dtype=np.int32)
    answer_len = np.asarray(answer_len, dtype=np.int32)
    answer_input_batch_bert=answer_input_batch_bert
    answer_attention_mask_batch_bert=answer_attention_mask_batch_bert
    answer_token_type_ids_batch_bert=answer_token_type_ids_batch_bert
    print(answer_encoded.shape)

    answer_bert_len = np.asarray(answer_bert_len, dtype=np.int32)

    glove_matrix = None
    glove_matrix_subject = None
    glove_matrix_relation = None
    glove_matrix_object = None
    if args.mode == 'train':
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}

        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []

        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)

    print('Writing', args.output_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
            'questions': questions_encoded,
            'question_input_bert': question_input_batch_bert,   
            'question_mask_bert': question_attention_mask_batch_bert,
            'question_ids_bert': question_token_type_ids_batch_bert,        
            'question_bert_len': question_bert_len,                          
            'questions_len': questions_len,
            'question_id': question_ids,
            'video_ids': video_ids_tbw,
            'video_names': video_names_tbw,
            'answers': all_answers,
            'answer': answer_encoded,
            'answer_input_bert': answer_input_batch_bert,   
            'answer_mask_bert': answer_attention_mask_batch_bert,
            'answer_ids_bert': answer_token_type_ids_batch_bert,        
            'answer_bert_len': answer_bert_len,                          
            'answer_len': answer_len,
            'answer_id': answer_ids,
            'ans_candidates': all_answer_cands_encoded,
            'ans_candidates_len': all_answer_cands_len,
            'ans_candidates_input_bert': all_candidateds_bert_encoded,   
            'ans_candidates_mask_bert': all_candidates_attention_mask_batch_bert,
            'ans_candidates_ids_bert': all_candidates_token_type_ids_batch_bert,        
            'ans_candidates_bert_len': all_candidates_bert_len,    
            'glove': glove_matrix,
            'questions_type' : questions_type,
        }
    with open(args.output_pt.format(args.dataset, args.dataset, args.mode), 'wb') as f:
        pickle.dump(obj, f)