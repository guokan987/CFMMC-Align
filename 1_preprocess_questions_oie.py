import argparse
import numpy as np
import os
import jsonlines
from preprocess.datautils import sutd_qa
import nltk
from openie import StanfordOpenIE
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
properties = {
    'openie.affinity_probability_cap': 2/3,
}

if __name__ == '__main__':
    nltk.download('punkt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sutd-qa', choices=['sutd-qa','tgif-qa', 'msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument('--answer_top', default=4000, type=int)
    parser.add_argument('--glove_pt', default="data/glove/glove.840.300d.pkl", type=str,
                        help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str, default='data/{}/{}_{}_questions_bert.pt')
    parser.add_argument('--output_subject_pt', type=str, default='data/{}/{}_{}_questions_subject_bert.pt')
    parser.add_argument('--output_relation_pt', type=str, default='data/{}/{}_{}_questions_relation_bert.pt')
    parser.add_argument('--output_object_pt', type=str, default='data/{}/{}_{}_questions_object_bert.pt')
    parser.add_argument('--vocab_json', type=str, default='data/{}/{}_vocab_bert.json')
    parser.add_argument('--vocab_subject_json', type=str, default='data/{}/{}_vocab_subject_bert.json')
    parser.add_argument('--vocab_relation_json', type=str, default='data/{}/{}_vocab_relation_bert.json')
    parser.add_argument('--vocab_object_json', type=str, default='data/{}/{}_vocab_object_bert.json')
    parser.add_argument('--mode', default='train',choices=['train', 'val', 'test'])
    parser.add_argument('--question_type', choices=['frameqa', 'action', 'transition', 'count', 'none'], default='none')
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset == 'sutd-qa':
        args.annotation_file = 'datasets/SUTD-TrafficQA/annotations/R2_{}.jsonl'.format(args.mode)
        # check if data folder exists
        if not os.path.exists('data/{}'.format(args.dataset)):
            os.makedirs('data/{}'.format(args.dataset))
        sutd_qa.process_questions_oie_keybert_bert3(args)#sutd_qa.process_questions_oie_keybert_bert2 for CMCIR
    