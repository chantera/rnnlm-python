%module rnnlm

%{
#include "../rnnlm/rnnlmlib.h"
%}

%include "std_string.i"
%include "std_vector.i"

namespace std {
  %template(StringVector) vector<string>;
}

%include ../rnnlm/rnnlmlib.h
