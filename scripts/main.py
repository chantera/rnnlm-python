#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is a python implementation of rnnlm.cpp
"""

import sys
import os.path
import argparse
import rnnlm


def main(argv):
    argc = len(argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug')
    parser.add_argument('-train')
    parser.add_argument('-one-iter', action='store_true')
    parser.add_argument('-max-iter')
    parser.add_argument('-valid')
    parser.add_argument('-nbest', action='store_true')
    parser.add_argument('-test')
    parser.add_argument('-class')
    parser.add_argument('-old-classes', action='store_true')
    parser.add_argument('-lambda')
    parser.add_argument('-gradient-cutoff')
    parser.add_argument('-dynamic')
    parser.add_argument('-gen')
    parser.add_argument('-independent', action='store_true')
    parser.add_argument('-alpha')
    parser.add_argument('-beta')
    parser.add_argument('-min-improvement')
    parser.add_argument('-anti-kasparek')
    parser.add_argument('-hidden')
    parser.add_argument('-compression')
    parser.add_argument('-direct')
    parser.add_argument('-direct-order')
    parser.add_argument('-bptt')
    parser.add_argument('-bptt-block')
    parser.add_argument('-rand-seed')
    parser.add_argument('-lm-prob')
    parser.add_argument('-binary', action='store_true')
    parser.add_argument('-rnnlm')
    args = parser.parse_args()

    debug_mode = 1

    fileformat = rnnlm.TEXT

    train_mode = 0
    valid_data_set = 0
    test_data_set = 0
    rnnlm_file_set = 0

    alpha_set = 0
    train_file_set = 0

    class_size = 100
    old_classes = 0
    lmda = 0.75
    gradient_cutoff = 15
    dynamic = 0
    starting_alpha = 0.1
    regularization = 0.0000001
    min_improvement = 1.003
    hidden_size = 30
    compression_size = 0
    direct = 0
    direct_order = 3
    bptt = 0
    bptt_block = 10
    gen = 0
    independent = 0
    use_lmprob = 0
    rand_seed = 1
    nbest = 0
    one_iter = 0
    max_iter = 32767  # 2^15-1
    anti_k = 0

    train_file = ""
    valid_file = ""
    test_file = ""
    rnnlm_file = ""
    lmprob_file = ""

    if argc == 1:
        # print("Help")

        print("Recurrent neural network based language modeling toolkit v 0.3d\n")

        print("Options:")

        print("Parameters for training phase:")

        print("\t-train <file>")
        print("\t\tUse text data from <file> to train rnnlm model")

        print("\t-class <int>")
        print("\t\tWill use specified amount of classes to decompose vocabulary; default is 100")

        print("\t-old-classes")
        print("\t\tThis will use old algorithm to compute classes, which results in slower models but can be a bit more precise")

        print("\t-rnnlm <file>")
        print("\t\tUse <file> to store rnnlm model")

        print("\t-binary")
        print("\t\tRnnlm model will be saved in binary format (default is plain text)")

        print("\t-valid <file>")
        print("\t\tUse <file> as validation data")

        print("\t-alpha <float>")
        print("\t\tSet starting learning rate; default is 0.1")

        print("\t-beta <float>")
        print("\t\tSet L2 regularization parameter; default is 1e-7")

        print("\t-hidden <int>")
        print("\t\tSet size of hidden layer; default is 30")

        print("\t-compression <int>")
        print("\t\tSet size of compression layer; default is 0 (not used)")

        print("\t-direct <int>")
        print("\t\tSets size of the hash for direct connections with n-gram features in millions; default is 0")

        print("\t-direct-order <int>")
        print("\t\tSets the n-gram order for direct connections (max %d); default is 3\n", rnnlm.MAX_NGRAM_ORDER)

        print("\t-bptt <int>")
        print("\t\tSet amount of steps to propagate error back in time; default is 0 (equal to simple RNN)")

        print("\t-bptt-block <int>")
        print("\t\tSpecifies amount of time steps after which the error is backpropagated through time in block mode (default 10, update at each time step = 1)")

        print("\t-one-iter")
        print("\t\tWill cause training to perform exactly one iteration over training data (useful for adapting final models on different data etc.)")

        print("\t-max-iter")
        print("\t\tWill cause training to perform exactly <max-iter> iterations over training data (useful to test static learning rates if min-improvement is set to 0.0)")

        print("\t-anti-kasparek <int>")
        print("\t\tModel will be saved during training after processing specified amount of words")

        print("\t-min-improvement <float>")
        print("\t\tSet minimal relative entropy improvement for training convergence; default is 1.003")

        print("\t-gradient-cutoff <float>")
        print("\t\tSet maximal absolute gradient value (to improve training stability, use lower values; default is 15, to turn off use 0)")

        print("Parameters for testing phase:")

        print("\t-rnnlm <file>")
        print("\t\tRead rnnlm model from <file>")

        print("\t-test <file>")
        print("\t\tUse <file> as test data to report perplexity")

        print("\t-lm-prob")
        print("\t\tUse other LM probabilities for linear interpolation with rnnlm model; see examples at the rnnlm webpage")

        print("\t-lambda <float>")
        print("\t\tSet parameter for linear interpolation of rnnlm and other lm; default weight of rnnlm is 0.75")

        print("\t-dynamic <float>")
        print("\t\tSet learning rate for dynamic model updates during testing phase; default is 0 (static model)")

        print("Additional parameters:")

        print("\t-gen <int>")
        print("\t\tGenerate specified amount of words given distribution from current model")

        print("\t-independent")
        print("\t\tWill erase history at end of each sentence (if used for training, this switch should be used also for testing & rescoring)")

        print("\nExamples:")
        print("python main.py -train train -rnnlm model -valid valid -hidden 50")
        print("python main.py -rnnlm model -test test")
        print("")

        return 0

    # print(args)

    # set debug mode
    if args.debug is not None:
        debug_mode = int(args.debug)
        if debug_mode > 0:
            print("debug mode: %d" % debug_mode)
    # else:
    #     print("ERROR: debug mode not specified!")
    #     return 0

    # search for train file
    if args.train is not None:
        train_file = args.train

        if debug_mode > 0:
            print("train file: %s" % train_file)

        if not os.path.isfile(train_file):
            print("ERROR: training data file not found!")
            return 0

        train_mode = 1
        train_file_set = 1
    # else:
    #     print("ERROR: training data file not specified!")
    #     return 0

    # set one-iter
    if args.one_iter:
        one_iter = 1

        if debug_mode > 0:
            print("Training for one iteration")

    # set max-iter
    if args.max_iter is not None:
        max_iter = int(args.max_iter)

        if debug_mode > 0:
            print("Maximum number of iterations: %d" % max_iter)
    # else:
    #     print("ERROR: maximum number of iterations not specified!")
    #     return 0

    # search for validation file
    if args.valid is not None:
        valid_file = args.valid

        if debug_mode > 0:
            print("valid file: %s" % valid_file)

        if not os.path.isfile(valid_file):
            print("ERROR: validation data file not found!")
            return 0

        valid_data_set = 1
    # else:
    #     print("ERROR: validation data file not specified!")
    #     return 0

    if train_mode and not valid_data_set:
        if one_iter == 0:
            print("ERROR: validation data file must be specified for training!")
            return 0

    # set nbest rescoring mode
    if args.nbest:
        nbest = 1

        if debug_mode > 0:
            print("Processing test data as list of nbests")

    # search for test file
    if args.test is not None:
        test_file = args.test

        if debug_mode > 0:
            print("test file: %s" % test_file)

        if nbest and test_file != "-":
            pass
        else:
            if not os.path.isfile(test_file):
                print("ERROR: test data file not found!")
                return 0

        test_data_set = 1
    # else:
    #     print("ERROR: test data file not specified!")
    #     return 0

    # set class size parameter
    if getattr(args, 'class') is not None:
        class_size = int(getattr(args, 'class'))

        if debug_mode > 0:
            print("class size: %d" % class_size)
    # else:
    #     print("ERROR: amount of classes not specified!")
    #     return 0

    # set old class
    if args.old_classes:
        old_classes = 1

        if debug_mode > 0:
            print("Old algorithm for computing classes will be used")

    # set lambda
    if getattr(args, 'lambda') is not None:
        lmda = float(getattr(args, 'lambda'))

        if debug_mode > 0:
            print("Lambda (interpolation coefficient between rnnlm and other lm): %f" % lmda)
    # else:
    #     print("ERROR: lambda not specified!")
    #     return 0

    # set gradient cutoff
    if args.gradient_cutoff is not None:
        gradient_cutoff = float(args.gradient_cutoff)

        if debug_mode > 0:
            print("Gradient cutoff: %f" % gradient_cutoff)
    # else:
    #     print("ERROR: gradient cutoff not specified!")
    #     return 0

    # set dynamic
    if args.dynamic is not None:
        dynamic = float(args.dynamic)

        if debug_mode > 0:
            print("Dynamic learning rate: %f" % dynamic)
    # else:
    #     print("ERROR: dynamic learning rate not specified!")
    #     return 0

    # set gen
    if args.gen is not None:
        gen = int(args.gen)

        if debug_mode > 0:
            print("Generating # words: %d" % gen)
    # else:
    #     print("ERROR: gen parameter not specified!")
    #     return 0

    # set independent
    if args.independent:
        independent = 1

        if debug_mode > 0:
            print("Sentences will be processed independently...")

    # set learning rate
    if args.alpha is not None:
        starting_alpha = float(args.alpha)

        if debug_mode > 0:
            print("Starting learning rate: %f" % starting_alpha)
        alpha_set = 1
    # else:
    #     print("ERROR: alpha not specified!")
    #     return 0

    # set regularization
    if args.beta is not None:
        regularization = float(args.beta)

        if debug_mode > 0:
            print("Regularization: %f" % regularization)
    # else:
    #     print("ERROR: beta not specified!n")
    #     return 0

    # set min improvement
    if args.min_improvement is not None:
        min_improvement = float(args.min_improvement)

        if debug_mode > 0:
            print("Min improvement: %f" % min_improvement)
    # else:
    #     print("ERROR: minimal improvement value not specified!")
    #     return 0

    # set anti kasparek
    if args.anti_kasparek is not None:
        anti_k = int(args.anti_kasparek)

        if anti_k != 0 and anti_k < 10000:
            anti_k = 10000

        if debug_mode > 0:
            print("Model will be saved after each # words: %d", anti_k)
    # else:
    #     print("ERROR: anti-kasparek parameter not set!")
    #     return 0

    # set hidden layer size
    if args.hidden is not None:
        hidden_size = int(args.hidden)

        if debug_mode > 0:
            print("Hidden layer size: %d" % hidden_size)
    # else:
    #     print("ERROR: hidden layer size not specified!")
    #     return 0

    # set compression layer size
    if args.compression is not None:
        compression_size = int(args.compression)

        if debug_mode > 0:
            print("Compression layer size: %d" % compression_size)
    # else:
    #     print("ERROR: compression layer size not specified!")
    #     return 0

    # set direct connections
    if args.direct is not None:
        direct = int(args.direct)

        direct = direct * 1000000
        if direct < 0:
            direct = 0

        if debug_mode > 0:
            print("Direct connections: %dM" % (int)(direct / 1000000))
    # else:
    #     print("ERROR: direct connections not specified!")
    #     return 0

    # set order of direct connections
    if args.direct_order is not None:
        direct_order = int(args.direct_order)
        if direct_order > rnnlm.MAX_NGRAM_ORDER:
            direct_order = rnnlm.MAX_NGRAM_ORDER

        if debug_mode > 0:
            print("Order of direct connections: %d" % direct_order)
    # else:
    #     print("ERROR: direct order not specified!")
    #     return 0

    # set bptt
    if args.bptt is not None:
        bptt = int(args.bptt)
        bptt = bptt + 1
        if bptt < 1:
            bptt = 1

        if debug_mode > 0:
            print("BPTT: %d" % (bptt - 1))
    # else:
    #     print("ERROR: bptt value not specified!")
    #     return 0

    # set bptt block
    if args.bptt_block is not None:
        bptt_block = int(args.bptt_block)
        if bptt_block < 1:
            bptt_block = 1

        if debug_mode > 0:
            print("BPTT block: %d" % bptt_block)
    # else:
    #     print("ERROR: bptt block value not specified!")
    #     return 0

    # set random seed
    if args.rand_seed is not None:
        rand_seed = int(args.rand_seed)

        if debug_mode > 0:
            print("Rand seed: %d" % rand_seed)
    # else:
    #     print("ERROR: Random seed variable not specified!")
    #     return 0

    # use other lm
    if args.lm_prob is not None:
        lmprob_file = args.lm_prob

        if debug_mode > 0:
            print("other lm probabilities specified in: %s" % lmprob_file)

        if not os.path.isfile(lmprob_file):
            print("ERROR: other lm file not found!")
            return 0

        use_lmprob = 1
    # else:
    #     print("ERROR: other lm file not specified!")
    #     return 0

    # search for binary option
    if args.binary:
        if debug_mode > 0:
            print("Model will be saved in binary format")

        fileformat = rnnlm.BINARY

    # search for rnnlm file
    if args.rnnlm is not None:
        rnnlm_file = args.rnnlm

        if debug_mode > 0:
            print("rnnlm file: %s" % rnnlm_file)

        rnnlm_file_set = 1
    # else:
    #     print("ERROR: model file not specified!")
    #     return 0

    if train_mode and not rnnlm_file_set:
        print("ERROR: rnnlm file must be specified for training!")
        return 0
    if test_data_set and not rnnlm_file_set:
        print("ERROR: rnnlm file must be specified for testing!")
        return 0
    if not test_data_set and not train_mode and gen == 0:
        print("ERROR: training or testing must be specified!")
        return 0
    if gen > 0 and not rnnlm_file_set:
        print("ERROR: rnnlm file must be specified to generate words!")
        return 0

    if train_mode:
        model1 = rnnlm.CRnnLM()

        model1.setTrainFile(train_file)
        model1.setRnnLMFile(rnnlm_file)
        model1.setFileType(fileformat)

        model1.setOneIter(one_iter)
        model1.setMaxIter(max_iter)
        if one_iter == 0:
            model1.setValidFile(valid_file)

        model1.setClassSize(class_size)
        model1.setOldClasses(old_classes)
        model1.setLearningRate(starting_alpha)
        model1.setGradientCutoff(gradient_cutoff)
        model1.setRegularization(regularization)
        model1.setMinImprovement(min_improvement)
        model1.setHiddenLayerSize(hidden_size)
        model1.setCompressionLayerSize(compression_size)
        model1.setDirectSize(direct)
        model1.setDirectOrder(direct_order)
        model1.setBPTT(bptt)
        model1.setBPTTBlock(bptt_block)
        model1.setRandSeed(rand_seed)
        model1.setDebugMode(debug_mode)
        model1.setAntiKasparek(anti_k)
        model1.setIndependent(independent)

        model1.alpha_set = alpha_set
        model1.train_file_set = train_file_set
        model1.trainNet()

    if test_data_set and rnnlm_file_set:
        model1 = rnnlm.CRnnLM()

        model1.setLambda(lmda)
        model1.setRegularization(regularization)
        model1.setDynamic(dynamic)
        model1.setTestFile(test_file)
        model1.setRnnLMFile(rnnlm_file)
        model1.setRandSeed(rand_seed)
        model1.useLMProb(use_lmprob)
        if use_lmprob:
            model1.setLMProbFile(lmprob_file)
        model1.setDebugMode(debug_mode)

        if nbest == 0:
            model1.testNet()
        else:
            model1.testNbest()

    if gen > 0:
        model1 = rnnlm.CRnnLM()

        model1.setRnnLMFile(rnnlm_file)
        model1.setDebugMode(debug_mode)
        model1.setRandSeed(rand_seed)
        model1.setGen(gen)

        model1.testGen()

    return 0


if __name__ == "__main__":
    main(sys.argv)
