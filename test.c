#include<stdio.h>
#include<math.h>

float data[1000];

void writeToArray(int rowNum, int col, int decrement, float*mat,float*vec);

//Sequence: size,row_num start, matrix element,vec
int main(){
    int row=4,col=3;
    int decrement = ceil(row/10.0);
    if(decrement==0) decrement=1;
    float vec[4]={1.0,2.0,3.0,4.0}, mat[16]={1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0};

    int rowNum=0;
    while(row-decrement>=0){
        row-=decrement;
        writeToArray(rowNum,col,decrement,mat,vec);
        rowNum+=decrement;
    }

    if(row>0){
        writeToArray(rowNum,col,row,mat,vec);
    }

    return 0;
}


void writeToArray(int rowNum, int col, int decrement, float*mat,float*vec){
    int index=0;
    data[index] = col*decrement+col+2;index++;
    data[index]= rowNum;index++;
    data[index]= col;index++;


    //Put in matrix
    for(int i=0;i<decrement;++i){
        int startPos = rowNum*col, endPos = startPos+col-1;
        while(startPos<=endPos){
            data[index]=mat[startPos];
            startPos++;index++;
        }
        rowNum++;
    }

    //Put in vec
    for(int i=0;i<col;++i){
        data[index]= vec[i];
        index++;
    }


    for(int i=0;i<index;++i){
        printf("%.2f ",data[i]);
    }
    printf("\n");
}