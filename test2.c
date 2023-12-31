#include <stdio.h>

float out[1000];

int main(){
    float mat[1000] = {14,0,3,1,2,3,4,5,6,8,9,10,1,2,3};

    float sum=0,matrix[1000],vec[1000];
    int id,size,col;

    size = mat[0];
    id = mat[1];
    col = mat[2];

    //reading
    int index=3;
    for(int i=0;i<size-col-2;++i){
        matrix[i]=mat[index];
        index++;
    }

    index=col-1;
    int i = size;
    while(index>=0){
        vec[index] = mat[i];
        index--;i--;
    }

    //calculate
    for(int i=0;i<(size-col-2)/col;++i){
        sum=0;
        for(int j=0;j<col;++j){
            sum+=matrix[i*col+j]*vec[j];
        }
        out[id] = sum;
        id++;
    }


     
    for(int i=0;i<20;++i){
        printf("%.2f\n",out[i]);
    }
    // for(int i=0;i<size-col-2;++i){
    //     printf("%.2f ",matrix[i]);
    // }
    // printf("\n");



    return 0;
}