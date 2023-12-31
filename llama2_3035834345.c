/*
PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME: llama2_3035834345.c
* NAME: Abel Haris Harsono
* UID : 3035834345
* Development Platform: WorkBench
* Remark: How much you implemented? Should be everything
* How to compile: gcc -o llama2_[UID] llama2_[UID].c utilities.c -O2 -pthread -lm

Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin

In compile, remember to add `-pthred` to link library:
$ gcc -o llama2_3035834345 llama2_3035834345.c utilities.c -O2 -pthread -lm

Then Run with:
$ ./llama2_[UID] <seed> <thr_count>
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"

/**
 * ----------------------------------------------------------------------------
 * TASK - Optimize Matrix-Vector Multiplication by Multi-Threading
 *
 * Matrix-Vector Multiplication, used in Attention and Feed-Forward Network
 * is the most time-consuming part of GPT. Luckily, most of computation is
 * independent of each row, so we can use Multi-Threading for acceleration.
 *
 * Please use <pthread.h> and your favorite synchronization method,
 * semaphore / mutex lock + conditional variable
 *
 * A sequential version is provided in seq.c, please modify it to parallel version.
 */

// YOUR CODE STARTS HERE

// Addtional Header File Here
#include <pthread.h>

// Global Variables
struct rusage main_usage, thread_usage[1000]; // get usage for main thread
int filled, param[1000], arg_glob[4], thread_count = 0, workDone,row_glob;
float *mat_glob, *out_glob, *vec_glob;


pthread_t threads[1000];
pthread_mutex_t accessLock, mv;
pthread_cond_t doneRunning, empty, full;
int finish=0;

// Function prototypes
void *thr_func(void *arg);
void readFromArray(int *start, int *id, int *decrement, int *col);
void writeToArray(int entry, int col, int decrement);
void calculate(int start, int id, int decrement, int col);

// create_mat_vec_mul()
int init_mat_vec_mul(int thr_count)
{
    thread_count = thr_count;

    // For sharing parameters to threads
    filled = 0;
    pthread_mutex_init(&mv, NULL);
    pthread_cond_init(&empty, NULL);
    pthread_cond_init(&full, NULL);

    // Putting things in output vector
    workDone = 0;
    pthread_mutex_init(&accessLock, NULL);
    pthread_cond_init(&doneRunning, NULL);

    for (int i = 0; i < thr_count; ++i)
    {
        param[i] = i;
        pthread_create(&threads[i], NULL, thr_func, &param[i]);
    }

    return 0;
}

#include <unistd.h>
void mat_vec_mul(float *out, float *vec, float *mat, int col, int row)
{
    //Setting parameters
    //---------------------------------------------------------------------
    out_glob = out;
    vec_glob = vec;
    mat_glob = mat;
    int decrement = ceil(row/(1.0 * thread_count));
    if(decrement==0) decrement=1;
    int row_temp  = row;
    row_glob = row;
    int rowNum=0;
    //Producer-consumer locks
    while(row-decrement>=0){
        row-=decrement;
        pthread_mutex_lock(&mv);
        while(filled){
            pthread_cond_wait(&empty,&mv);
        }
        writeToArray(rowNum,col,decrement);
        filled=1;
        pthread_cond_signal(&full);
        pthread_mutex_unlock(&mv);
        rowNum+=decrement;
    }
    if(row>0){
        pthread_mutex_lock(&mv);
        while(filled){
            pthread_cond_wait(&empty,&mv);
        }
        writeToArray(rowNum,col,row);
        filled=1;
        pthread_cond_signal(&full);
        pthread_mutex_unlock(&mv);
    }
    //---------------------------------------------------------------------


    //Waiting for everyone to finish
    pthread_mutex_lock(&accessLock);
    while(workDone<row_temp){
        pthread_cond_wait(&doneRunning, &accessLock);
    }
    // printf("Things put in out: %d\n",row_temp);
    pthread_mutex_unlock(&accessLock);
    // printf("filled: %d\n",filled);
    // printf("workDone: %d, row_temp: %d\n",workDone,row_temp);
    workDone=0;filled=0;
}


int close_mat_vec_mul()
{
    finish=1;
    pthread_cond_broadcast(&full);
    pthread_mutex_destroy(&mv);
    pthread_mutex_destroy(&accessLock);
    pthread_cond_destroy(&full);
    pthread_cond_destroy(&empty);
    pthread_cond_destroy(&doneRunning);
    for (int i = 0; i < thread_count; ++i)
    {
        struct rusage *rthread;
        void * temp;
        pthread_join(threads[i],&temp);
        rthread = (struct rusage*)temp;
        if(rthread!=NULL)
        printf("thread %d has completed - user: %.4f s, system: %.4f s\n", i,
            (rthread->ru_utime.tv_sec + rthread->ru_utime.tv_usec/1000000.0),
            (rthread->ru_stime.tv_sec + rthread->ru_utime.tv_usec/1000000.0));
    }
    getrusage(RUSAGE_SELF, &main_usage);
    printf("main thread - user: %.4f s, system: %.4f s\n",
    (main_usage.ru_utime.tv_sec + main_usage.ru_utime.tv_usec/1000000.0),
    (main_usage.ru_stime.tv_sec + main_usage.ru_stime.tv_usec/1000000.0));
    return 0;
}

//The equivalent of a consumer
void *thr_func(void *arg)
{
    int *num = (int *)arg, start, id, decrement, col;
    while (1)
    {
        pthread_mutex_lock(&mv);
        while (!filled)
        {       
            pthread_cond_wait(&full, &mv);
            if(finish==1){
                getrusage(RUSAGE_THREAD,&thread_usage[*num]);
                pthread_mutex_unlock(&mv);
                return (void *)&thread_usage[*num];
            }
        }
        filled = 0;
        readFromArray(&start, &id, &decrement, &col);
        pthread_cond_signal(&empty);
        pthread_mutex_unlock(&mv);

        calculate(start, id, decrement,col);
        if(workDone == row_glob){
            pthread_cond_signal(&doneRunning);
        }
    }
    return num;
}

//Setting parameters for consumers
void writeToArray(int entry, int col, int decrement)
{
    arg_glob[0] = entry;
    arg_glob[1] = entry*col; //Where we're starting
    arg_glob[2] = decrement; //Number of rows taken care of by thread
    arg_glob[3] = col;
}

void readFromArray(int *start, int *id, int *decrement, int *col)
{
    *id = arg_glob[0];
    *start = arg_glob[1];
    *decrement = arg_glob[2];
    *col = arg_glob[3];
}

void calculate(int start, int id, int decrement, int col)
{
    float sum;
    // int count = 0;
    for (int i = 0; i < decrement; ++i)
    {
        sum = 0;
        for (int j = 0; j < col; ++j)
        {
            sum += mat_glob[start+j] * vec_glob[j];
        }
        start+=col;
        // count++;
        out_glob[id] = sum;
         id++;
    }
    pthread_mutex_lock(&accessLock);
        
       
        workDone+=decrement;
        pthread_mutex_unlock(&accessLock);
    
}

// For testing
// int main(){
//     int row=4,col=4;
//     float vec[4]={1.0,2.0,3.0,4.0}, mat[16]={1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0},out[4];
//     init_mat_vec_mul(2);
//     mat_vec_mul(out,vec,mat,col,row);
//     printf("Results:\n");
//     for(int i=0;i<row;++i){
//         printf("%.2f\n",out[i]);
//     }

//     float out1[4],vec1[]={5,6,7,8};
//     mat_vec_mul(out1,vec1,mat,col,row);
//     printf("Results1:\n");
//     for(int i=0;i<row;++i){
//         printf("%.2f\n",out1[i]);
//     }

//     float out2[4],vec2[]={100,10,11,12};
//     mat_vec_mul(out2,vec2,mat,col,row);
//     printf("Results2:\n");
//     for(int i=0;i<row;++i){
//         printf("%.2f\n",out2[i]);
//     }

//     return 0;
// }

// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig *p, LLMRuntime *s, LLMWeight *w)
{

    // a few convenience variables
    int dim = p->dim, hidden_dim = p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim * sizeof(float));

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++)
    {
        // printf("Looped!\n");

        // Attention
        {
            //  printf("Entering\n");
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l * dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l * dim * dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l * dim * dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);

            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
           

            // residual connection back into x
            accum(s->x, s->xb2, dim);
            //  printf("Exited\n");
        }

        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            //  printf("Entering2\n");
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l * dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
            
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l * dim * hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
            // printf("Exited2\n");
        }
    }
  

    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
      
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q = 0; q < p->vocab_size; q++)
    {
        s->logits[q] /= 0.9f;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char *argv[])
{

    unsigned int seed;
    int thr_count;

    if (argc == 3)
    {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    }
    else
    {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    init_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1)
    {
        return 1;
    }

    // load tokenizer
    char **vocab = malloc(config.vocab_size * sizeof(char *));
    if (load_tokenizer(vocab, config.vocab_size) == 1)
    {
        return 1;
    }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);

    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len)
    {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end - start) / 1000, config.seq_len / (double)(end - start) * 1000);

    // cleanup
    close_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++)
    {
        free(vocab[i]);
    }
    free(vocab);
    return 0;
}