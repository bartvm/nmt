#!/usr/bin/env python

import argparse

def process():

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_sentences",dest="src_data",required=True,metavar="FILE")
    parser.add_argument("--target_sentences",dest="trg_data",required=True,metavar="FILE")
    parser.add_argument("--n_epoch",dest="n_epoch",default=10,type=int,metavar="INT",help="Number of iterations over the training set")
    parser.add_argument("--decay_rate",dest="decay_rate",default=1.2,type=float,metavar="FLOAT",help="")
    parser.add_argument("--decay_after",dest="decay_after",default=10,type=int,metavar="INT",help="")
    parser.add_argument("--burnin",dest="burnin",default=0,type=int,metavar="INT",help="Burn-in epochs")
    parser.add_argument("--thinning",dest="thinning",default=100,type=int,metavar="INT",help="Thinning interval")
    parser.add_argument("--prior_prec",dest="prior_prec",default=1,type=float,metavar="FLOAT",help="Precision of prior normal distribution")
    parser.add_argument("--g_clip",dest="g_clip",default=1.0,type=float,metavar="FLOAT",help="Constraint on the maximum norm of gradients")
    parser.add_argument("--sgmcmc",dest="sgmcmc",default="sgld",metavar="STR",help="Type of Stochastic Gradient Monte Carlo")
    args = parser.parse_args()
    
    
