/*-
 * Copyright (c) 2007 Charl Linssen <reverse moc.sdribgnirut@lrahc>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Names of the copyright holders must not be used to endorse or promote
 *    products derived from this software without prior written permission 
 *    from the copyright holders.
 * 4. If any files are modified, you must cause the modified files to carry
 *    prominent notices stating that you changed the files and the date of
 *    any change.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>

// following flag is for printing debug information
#define DEBUG

// Activation function, and its derivative applied to the net output
#define ACTFUNC(x)      (1.0 / (1.0 + exp(-1.0 * (x))))
#define ACTFUNCDER(x)   ((x) * (1 - (x)))

#define sqr(x)          ((x) * (x))


const int max_epoch = 10000;
const float learning_rate = 0.25;


// patterns
int pat_width = 0;
int pat_height = 0;
int nr_pats = 0;
int nr_unique_pats = 0;
int ***patterns; // array, indexed as [x][y][pattern_index]
int *pattern_class; // array, stores what class each pattern belongs to

// network topology
int nr_input, nr_hidden, nr_output;
float **weights_input_hidden, **weights_hidden_output;
float *bias_hidden, *bias_output;

// scratch variables for propagating signals
float *inputs;
float *output_hidden;
float *output_output;


// this function generates a random number in the range [0..range]
int randomrange(int range)
{
  float r = (rand() / (float)RAND_MAX);
  r *= range;
  return (int)round(r);
}
	
void initialize(const int nr_hidden_)
{
  int i, j, o, h;
  
  nr_input = pat_width * pat_height;
  nr_hidden = nr_hidden_;
  nr_output = nr_unique_pats;
  
  inputs = malloc(sizeof(float) * nr_input);
  output_hidden = malloc(sizeof(float) * nr_hidden);
  output_output = malloc(sizeof(float) * nr_output);

  #ifdef DEBUG
  printf("Network topology: %d input units; %d hidden; %d output\n", nr_input, nr_hidden, nr_output);
  #endif

  for (i=0; i < nr_input; ++i)
    inputs[i] = 0.0;
  for (h=0; h < nr_hidden; ++h)
    output_hidden[h] = 0.0;
  for (o=0; o < nr_output; ++o)
    output_output[o] = 0.0;

  weights_input_hidden = malloc(sizeof(float*) * nr_input);
  for (i=0; i < nr_input; ++i)
    weights_input_hidden[i] = malloc(sizeof(float) * nr_hidden);

  weights_hidden_output = malloc(sizeof(float*) * nr_hidden);
  for (h=0; h < nr_hidden; ++h)
    weights_hidden_output[h] = malloc(sizeof(float) * nr_output);
  
  bias_hidden = malloc(sizeof(float) * nr_hidden);
  bias_output = malloc(sizeof(float) * nr_output);

  for (h=0; h < nr_hidden; ++h)
    for (i=0; i < nr_input; ++i)
      weights_input_hidden[i][h] = randomrange(200)/1000.0 - 0.1;
  for (o=0; o < nr_output; ++o)
    for (h=0; h < nr_hidden; ++h)
      weights_hidden_output[h][o] = randomrange(200)/1000.0 - 0.1;
  
  for (h=0; h < nr_hidden; ++h)
    bias_hidden[h] = randomrange(200)/1000.0 - 0.1;
  for (o=0; o < nr_output; ++o)
    bias_output[o] = randomrange(200)/1000.0 - 0.1;

  // patterns array is initialized on reading input file
}

void propagate(int patIndex)
{
  if (patIndex >= nr_pats || patIndex < 0)
    return;

  int i, h, o;
  int r, c;
  float temp;

  #ifdef DEBUG
  printf("  == PROPAGATING NETWORK (pattern %d) ==========================\n", patIndex);
  #endif

  // put the inputs in a linear format (from two to one dimension)
  for (c=0; c < pat_width; ++c)
    for (r=0; r < pat_height; ++r)
      inputs[r*pat_width + c] = 1.0 * (float)patterns[c][r][patIndex];

  #ifdef DEBUG
  printf("  ==  Inputs: [");
  for (i=0; i < nr_input; ++i)
    printf("%f, ", inputs[i]);
  printf("]\n");
  printf("  ==  Weights from input to hidden:\n");
  for (h=0; h < nr_hidden; ++h)
  {
    for (i=0; i < nr_input; ++i)
      printf("  ==  Weight from input neuron %d to hidden neuron %d = %f\n", i, h, weights_input_hidden[i][h]);
    printf("  ==  Hidden neuron %d has bias %f\n", h, bias_hidden[h]);
  }
  #endif

  // calculate hidden layer outputs
  for (h=0; h < nr_hidden; ++h)
  {
    temp = 0.0;
    for (i=0; i < nr_input; ++i)
      temp += weights_input_hidden[i][h] * inputs[i];
    temp += bias_hidden[h];
    output_hidden[h] = ACTFUNC(temp);
  }

  #ifdef DEBUG
  printf("  == Outputs of hidden layer:\n  == ");
  for (i=0; i < nr_hidden; ++i)
    printf("   %f, ", output_hidden[i]);
  printf("\n  == Weights from hidden to output:\n");
  for (o=0; o < nr_output; ++o)
  {
    for (h=0; h < nr_hidden; ++h)
      printf("  ==  Weight from hidden neuron %d to output neuron %d = %f\n", h, o, weights_hidden_output[h][o]);
    printf("  ==  Output neuron %d has bias %f\n", o, bias_output[o]);
  }
  #endif

  // calculate output layer outputs
  for (o=0; o < nr_output; ++o)
  {
    temp = 0.0;
    for (h=0; h < nr_hidden; ++h)
      temp += weights_hidden_output[h][o] * output_hidden[h];
    temp += bias_output[o];
    output_output[o] = ACTFUNC(temp); 
  }

  #ifdef DEBUG
  float targets[nr_output];
  for (o=0; o < nr_output; ++o)
    targets[o] = 0.2;
  targets[pattern_class[patIndex]] = 0.8;

  printf("\n  == Targets: \n  == ");
  for (o=0; o < nr_output; ++o)
    printf("    %f, ", targets[o]);
  printf("\n  == Outputs of output layer:\n  == ");
  for (i=0; i < nr_output; ++i)
    printf("    %f, ", output_output[i]);
  printf("\n  =============================================\n");
  #endif
}

int train(const float max_error)
{
  float error = max_error + 1.0;
  float prev_error = max_error + 1.0;
  float temp;
  float targets[nr_output];
  float output_error[nr_output];
  float hidden_error[nr_hidden];
  int epoch = 0;
  int i, h, o, pat;
  
  while (error > max_error && epoch < max_epoch)
  {
    ++epoch;

    #ifdef DEBUG_
    printf("\n######### Training network (epoch %d, err %f) ###############\n", epoch, error);
    #endif

    prev_error = error;
    error = 0.0;

    for (pat=0; pat < nr_pats; ++pat)
    {
      propagate(pat);

      // establish targets (0.2 instead of 0 and 0.8 instead of 1 to keep in
      // somewhat linear part of sigmoid curve)
      for (o=0; o < nr_output; ++o)
        targets[o] = 0.2;
      targets[pattern_class[pat]] = 0.8;

      // calculate network error (sum of squares)
      temp = 0.0;
      for (o=0; o < nr_output; ++o)
        temp += sqr(targets[o] - output_output[o]);
      error = temp * 0.5;

      #ifdef DEBUG
      printf("  Targets: ");
      for (o=0; o < nr_output; ++o)
        printf("%f ", targets[o]);
      printf("\n");
      #endif

      // calculate output layer error
      for (o=0; o < nr_output; ++o)
        output_error[o] = (targets[o] - output_output[o]) *
                          ACTFUNCDER(output_output[o]);
      
      #ifdef DEBUG
      printf("  Output neuron errors: ");
      for (o=0; o < nr_output; ++o)
        printf("%f ", output_error[o]);
      printf("\n");
      #endif
      
      // calculate hidden layer error
      for (h=0; h < nr_hidden; ++h)
      {
        temp = 0.0;
        for (o=0; o < nr_output; ++o)
          temp += output_error[o] * weights_hidden_output[h][o];
        hidden_error[h] = temp * ACTFUNCDER(output_hidden[h]);
      }
      
      #ifdef DEBUG
      for (h=0; h < nr_hidden; ++h)
        printf("  hidden neuron %d has error %f\n", h, hidden_error[h]);
      #endif

      // update output weights
      for (o=0; o < nr_output; ++o)
        for (h=0; h < nr_hidden; ++h)
          weights_hidden_output[h][o] += learning_rate * output_error[o] *
                                         output_hidden[h];

      // update output bias
      for (o=0; o < nr_output; ++o)
        bias_output[o] += learning_rate * output_error[o];

      #ifdef DEBUG
      printf("  New output weights:\n");
      for (o=0; o < nr_output; ++o)
        for (h=0; h < nr_hidden; ++h)
          printf("    Output neuron %d has weight %d = %f\n", o, h, 
                                                   weights_hidden_output[h][o]);
      #endif

      // update hidden weights
      for (h=0; h < nr_hidden; ++h)
        for (i=0; i < nr_input; ++i)
          weights_input_hidden[i][h] += learning_rate * hidden_error[h] *
                                        inputs[i];

      // update hidden bias
      for (h=0; h < nr_hidden; ++h)
        bias_hidden[h] += learning_rate * hidden_error[h];
    }

    if (error < max_error)
      return 1;
      
    // The second part of the if-statement below is a hack to not have to use
    // a momentum term. It disallows the network from stopping training if it
    // hasn't iterated for enough epochs.
    if (error > prev_error && epoch > 10000)
    {
      printf("Error is going back up -> minimum found (err = %f), but not within range.\n", error);
      fprintf(stderr, "Error is going back up -> minimum found (err = %f), but not within range.\n", error);
      return 0;
    }
  }
  
  #ifdef DEBUG
  printf("##################################################\n\n");
  #endif
  
  if (error < max_error)
    return 1;

  return 0;
}

// The following function returns the number of the output neuron with highest
// confidence. It assumes that the network has been propagated.
// Returns integer in [0..nr_output].
int winner()
{
  int highest_index = 0;
  float highest_confidence = -1.0;
  int o;
  
  for (o=0; o < nr_output; ++o)
    if (output_output[o] > highest_confidence)
    {
      highest_confidence = output_output[o];
      highest_index = o;
    }

  return highest_index;
}

void make_confusion_matrix(FILE *testfile)
{
  int confusion_matrix[nr_unique_pats][nr_unique_pats]; // [predicted][actual]
  int i, j, pat;
  int nr_wrong = 0;

  for (i=0; i < nr_unique_pats; ++i)
    for (j=0; j < nr_unique_pats; ++j)
      confusion_matrix[i][j] = 0;

  for (pat=0; pat < nr_pats; ++pat)
  {
    propagate(pat);
    ++(confusion_matrix[pattern_class[pat]][winner()]);
    if (pattern_class[pat] != winner())
      ++nr_wrong;
  }

  // print
  fprintf(stderr, "-> Got %d wrong\n", nr_wrong);
  printf("\n\nConfusion matrix (horiz = network result; vert = actual class):\n");

  for (i=-1; i < nr_unique_pats; ++i)
  {
    for (j=-1; j < nr_unique_pats; ++j)
    {
      if (i < 0 && j >= 0)
      {
        printf("%d\t", j);
        if (j == nr_unique_pats - 1) printf("\n========+=========================================================================");
      }
      else if (i < 0)
        printf("\t| ");
      else if (j < 0)
        printf("%d\t| ", i);
      else
        printf("%d\t", confusion_matrix[i][j]);
    }
    printf("\n");
  }
}

/* This function will read patterns of the form:

     patwidth
     patheight
     nr_unique_patterns
     nr_patterns
     0
     X.
     .X
     1
     .X
     .X

   where  0 and 1 are class IDs
          . is whitespace (use a dot, not a space)

   This function may segfault/do odd things when input file has errors.
*/
void read_patterns(FILE *infile)
{
  int i, line, ch;
  char *s;

  fscanf(infile, "%d", &pat_width);
  fscanf(infile, "%d", &pat_height);
  fscanf(infile, "%d", &nr_unique_pats);
  fscanf(infile, "%d", &nr_pats);

  // allocate memory for patterns
  // ASSUMING ALLOCATION SUCCEEDS (this is bad)
  patterns = malloc(sizeof(int**) * pat_width);
  pattern_class = malloc(sizeof(int) * nr_pats);
  for (i=0; i < pat_width; ++i)
  {
    patterns[i] = malloc(sizeof(int*) * pat_height);
    for (ch=0; ch < pat_height; ++ch)
      patterns[i][ch] = malloc(sizeof(int) * nr_pats);
  }
  s = malloc(sizeof(char) * pat_width);

  #ifdef DEBUG
  printf("Pattern W/H/count/unique: %d/%d/%d/%d\n", pat_width,pat_height,nr_pats,nr_unique_pats);
  #endif

  // read patterns
  for (i = 0; i < nr_pats; ++i)
  {
    fscanf(infile, "%d", &(pattern_class[i]));
    #ifdef DEBUG
    printf("Pattern class of pattern #%d is %d\n", i, pattern_class[i]);
    #endif

    for (line = 0; line < pat_height; ++line)
    {
      fscanf(infile, "%s", s);

      #ifdef DEBUG
      printf(" Line %d: [%s]\n", line, s);
      #endif
      
      for (ch = 0; ch < pat_width; ++ch)
        if (s[ch] == 'X')
          patterns[ch][line][i] = 1;
        else
          patterns[ch][line][i] = 0;
    }
  }
  
  //TOBEREMOVED::
  printf("final pattern classes: ");
  for (i=0; i < nr_pats; ++i)
    printf("%d ", pattern_class[i]);
  printf("\n");
  
}

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    printf("Usage: nn_sim INFILE [TESTFILE]\n");
    printf("Make sure pattern dimensions are the same in both files.\n");
    return 1;
  }

  FILE *infile = fopen(argv[1], "r");
  if (infile == NULL)
  {
    fprintf(stderr, "Error: could not open input file.\n");
    exit(1);
  }

  FILE *testfile = NULL;
  int i,j,k;

  if (argc > 2)
    testfile = fopen(argv[2], "r");

  read_patterns(infile);

  fprintf(stderr, "Initializing at %d..\n", i);
  initialize(5);

  if (train(0.0001))
    printf("Training successful!\n");
  else
    printf("Training unsuccessful.\n");

  // evaluate performance
  if (testfile != NULL)
  {
    read_patterns(testfile);
    make_confusion_matrix(testfile);
  }


  return 0;
}


