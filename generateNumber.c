#include<stdio.h>

int main(){
    FILE *file = fopen("num.txt","w");
    for(int i=0;i<288*288;++i){
        fprintf(file,"%d, ",i+1);
    }
    fclose(file);
    return 0;
}