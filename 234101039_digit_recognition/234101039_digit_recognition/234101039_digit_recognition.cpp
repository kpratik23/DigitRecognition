// 234101039_digit_recognition.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include<stdio.h>
#include<string.h>
#include<limits.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<float.h>
#include<Windows.h>

#define K 32					
#define DELTA 0.00001			
#define EPSILON 0.03			
#define UNIVERSE_SIZE 50000		
#define CLIP 5000				
#define FS 320					
#define Q 12					
#define P 12					
#define pie (22.0/7)
#define N 5						
#define M 32					
#define T_ 400					
#define TRAIN_SIZE 20			
#define TEST_SIZE 50			

//HMM Model Variables
long double A[N + 1][N + 1],B[N + 1][M + 1], pi[N + 1], alpha[T_ + 1][N + 1], beta[T_ + 1][N + 1], gamma[T_ + 1][N + 1], delta[T_+1][N+1], xi[T_+1][N+1][N+1], A_bar[N + 1][N + 1],B_bar[N + 1][M + 1], pi_bar[N + 1];
int O[T_+1], q[T_+1], psi[T_+1][N+1], q_star[T_+1];
long double P_star=-1, P_star_dash=-1;

int samples[50000];
int T=160;
int start_frame;
int end_frame;

long double R[P+1];
long double a[P+1];
long double C[Q+1];
long double reference[M+1][Q+1];
long double tokhuraWeight[Q+1]={0.0, 1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};
long double energy[T_]={0};
long double X[UNIVERSE_SIZE][Q];
int LBG_M=0;
long double codebook[K][Q];
int cluster[UNIVERSE_SIZE];


void normalize_data(char file[100]){
	FILE* fp=fopen(file,"r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return;
	}
	int amp=0,avg=0;
	int i=0;
	int n=0;
	int min_amp=INT_MAX;
	int max_amp=INT_MIN;
	while(!feof(fp)){
		fscanf(fp,"%d",&amp);
		avg+=amp;
		min_amp=(amp<min_amp)?amp:min_amp;
		max_amp=(amp>max_amp)?amp:max_amp;
		n++;
	}
	avg/=n;
	T=(n-FS)/80 + 1;
	if(T>T_) T=T_;
	min_amp-=avg;
	max_amp-=avg;
	fseek(fp,0,SEEK_SET);
	while(!feof(fp)){
		fscanf(fp,"%d",&amp);
		if(min_amp==max_amp){
			amp=0;
		}
		else{
			amp-=avg;
			amp=(amp*CLIP)/((max_amp>min_amp)?max_amp:(-1)*min_amp);
			samples[i++]=amp;
		}
	}
	fclose(fp);
}

void calculate_energy_of_frame(int frame_no){
	int sample_start_index=frame_no*80;
	energy[frame_no]=0;
	for(int i=0;i<FS;i++){
		energy[frame_no]+=samples[i+sample_start_index]*samples[i+sample_start_index];
		energy[frame_no]/=FS;
	}
}

long double calculate_max_energy(){
	int nf=T;
	long double max_energy=DBL_MIN;
	for(int f=0;f<nf;f++){
		if(energy[f]>max_energy){
			max_energy=energy[f];
		}
	}
	return max_energy;
}

long double calculate_avg_energy(){
	int nf=T;
	long double avg_energy=0.0;
	for(int f=0;f<nf;f++){
		avg_energy+=energy[f];
	}
	return avg_energy/nf;
}

void mark_checkpoints(){
	int nf=T;
	//Calculate energy of each frame
	for(int f=0;f<nf;f++){
		calculate_energy_of_frame(f);
	}
	long double threshold_energy=calculate_avg_energy()/10;
	int isAboveThresholdStart=1;
	int isAboveThresholdEnd=1;
	start_frame=0;
	end_frame=nf-1;
	for(int f=0;f<nf-5;f++){
		for(int i=0;i<5;i++){
			isAboveThresholdStart*=(energy[f+i]>threshold_energy);
		}
		if(isAboveThresholdStart){
			start_frame=((f-5) >0)?(f-5):(0);
			break;
		}
		isAboveThresholdStart=1;
	}
	for(int f=nf-1;f>4;f--){
		for(int i=0;i<5;i++){
			isAboveThresholdEnd*=(energy[f-i]>threshold_energy);
		}
		if(isAboveThresholdEnd){
			end_frame=((f+5) < nf)?(f+5):(nf-1);
			break;
		}
		isAboveThresholdEnd=1;
	}
}

void load_codebook(){
	FILE* fp;
	fp=fopen("234101039_codebook.csv","r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return;
	}
	for(int i=1;i<=M;i++){
		for(int j=1;j<=Q;j++){
			fscanf(fp,"%Lf,",&reference[i][j]);
		}
	}
	fclose(fp);
}

void durbinAlgo(){
	long double E=R[0];
	long double alpha[13][13];
	for(int i=1;i<=P;i++){
		double k;
		long double numerator=R[i];
		long double alphaR=0.0;
		for(int j=1;j<=(i-1);j++){
			alphaR+=alpha[j][i-1]*R[i-j];
		}
		numerator-=alphaR;
		k=numerator/E;
		alpha[i][i]=k;
		for(int j=1;j<=(i-1);j++){
			alpha[j][i]=alpha[j][i-1]-(k*alpha[i-j][i-1]);
			if(i==P){
				a[j]=alpha[j][i];
			}
		}
		E=(1-k*k)*E;
		if(i==P){
			a[i]=alpha[i][i];
		}
	}
}

void autoCorrelation(int frame_no){
	long double s[FS];
	int sample_start_index=frame_no*80;
	
	for(int i=0;i<FS;i++){
		long double wn=0.54-0.46*cos((2*(22.0/7.0)*i)/(FS-1));
		s[i]=wn*samples[i+sample_start_index];
	}
	
	for(int i=0;i<=P;i++){
		long double sum=0.0;
		for(int y=0;y<=FS-1-i;y++){
			sum+=((s[y])*(s[y+i]));
		}
		R[i]=sum;
	}

	durbinAlgo();
}


void cepstralTransformation(){
	C[0]=2.0*(log(R[0])/log(2.0));
	for(int m=1;m<=P;m++){
		C[m]=a[m];
		for(int k=1;k<m;k++){
			C[m]+=((k*C[k]*a[m-k])/m);
		}
	}
}

void raisedSineWindow(){
	for(int m=1;m<=P;m++){
		long double wm=(1+(Q/2)*sin(pie*m/Q));
		C[m]*=wm;
	}
}

void process_universe_file(FILE* fp, char file[]){
	normalize_data(file);
	int m=0;
	int nf=T;
	for(int f=0;f<nf;f++){
		autoCorrelation(f);
		cepstralTransformation();
		raisedSineWindow();
		for(int i=1;i<=Q;i++){
			fprintf(fp,"%Lf,",C[i]);
		}
		fprintf(fp,"\n");
	}
}

void generate_universe(){
	int cnt=0;
	FILE* universefp;
	universefp=fopen("234101039_universe.csv","w");
	for(int d=0;d<=9;d++){
		for(int u=1;u<=TRAIN_SIZE;u++){
			char filename[39];
			_snprintf(filename,39,"234101039_dataset/234101039_E_%d_%d.txt",d,u);
			process_universe_file(universefp,filename);
		}
	}
}

int minTokhuraDistance(long double testC[]){
	long double minD=DBL_MAX;
	int minDi=0;
	for(int i=1;i<=M;i++){
		long double distance=0.0;
		for(int j=1;j<=Q;j++){
			distance+=(tokhuraWeight[j]*(testC[j]-reference[i][j])*(testC[j]-reference[i][j]));
		}
		if(distance<minD){
			minD=distance;
			minDi=i;
		}
	}
	return minDi;
}

void generate_observation_sequence(char file[]){
	FILE* fp=fopen("o.txt","w");
	normalize_data(file);
	int m=0;
	mark_checkpoints();
	T=(end_frame-start_frame+1);
	int nf=T;
	for(int f=start_frame;f<=end_frame;f++){
		autoCorrelation(f);
		cepstralTransformation();
		raisedSineWindow();
		fprintf(fp,"%d ",minTokhuraDistance(C));
	}
	fprintf(fp,"\n");
	fclose(fp);
}


void load_universe(char file[100]){
	FILE* fp=fopen(file,"r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return;
	}
	
	int i=0;
	long double c;
	while(!feof(fp)){
		fscanf(fp,"%Lf,",&c);
		X[LBG_M][i]=c;
		i=(i+1)%12;
		if(i==0) LBG_M++;
	}
	fclose(fp);
}


void store_codebook(char file[100],int k){
	FILE* fp=fopen(file,"w");
	if(fp==NULL){
		printf("Error opening file!\n");
		return;
	}
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			fprintf(fp,"%Lf,",codebook[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}



void print_codebook(int k){
	printf("Codebook of size %d:\n",k);
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			printf("%Lf\t",codebook[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}



void initialize_with_centroid(){
	long double centroid[12]={0.0};
	for(int i=0;i<LBG_M;i++){
		for(int j=0;j<12;j++){
			centroid[j]+=X[i][j];
		}
	}
	for(int i=0;i<12;i++){
		centroid[i]/=LBG_M;
		codebook[0][i]=centroid[i];
	}
}



long double calculate_distance(long double x[12], long double y[12]){
	long double distance=0.0;
	for(int i=0;i<12;i++){
		distance+=(tokhuraWeight[i+1]*(x[i]-y[i])*(x[i]-y[i]));
	}
	return distance;
}



void nearest_neighbour(int k){
	for(int i=0;i<LBG_M;i++){
		long double nn=DBL_MAX;
		int cluster_index;
		for(int j=0;j<k;j++){
			long double dxy=calculate_distance(X[i],codebook[j]);
			if(dxy<=nn){
				cluster_index=j;
				nn=dxy;
			}
		}
		cluster[i]=cluster_index;
	}
}


void codevector_update(int k){
	long double centroid[K][12]={0.0};
	int n[K]={0};
	for(int i=0;i<LBG_M;i++){
		for(int j=0;j<12;j++){
			centroid[cluster[i]][j]+=X[i][j];
		}
		n[cluster[i]]++;
	}
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			codebook[i][j]=centroid[i][j]/n[i];
		}
	}
}


long double calculate_distortion(){
	long double distortion=0.0;
	for(int i=0;i<LBG_M;i++){
		distortion+=calculate_distance(X[i],codebook[cluster[i]]);
	}
	distortion/=LBG_M;
	return distortion;
}



void KMeans(int k){
	FILE* fp=fopen("distortion.txt","a");
	if(fp==NULL){
		printf("Error pening file!\n");
		return;
	}
	int m=0;
	long double prev_D=DBL_MAX, cur_D=DBL_MAX;
	do{
		nearest_neighbour(k);
		m++;
		codevector_update(k);
		prev_D=cur_D;
		cur_D=calculate_distortion();
		fprintf(fp,"%Lf\n",cur_D);
	}while((prev_D-cur_D)>DELTA);
	fclose(fp);
}



void LBG(){
	int k=1;
	initialize_with_centroid();
	while(k!=K){
		for(int i=0;i<k;i++){
			for(int j=0;j<12;j++){
				long double Yi=codebook[i][j];
				codebook[i][j]=Yi-EPSILON;
				codebook[i+k][j]=Yi+EPSILON;
			}
		}
		k=k*2;
		KMeans(k);
	}
}

void generate_codebook(){
	load_universe("234101039_universe.csv");
	LBG();
	store_codebook("234101039_codebook.csv",K);
}

void initialization()
{
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			A[i][j] = 0;
		}
		for (int j = 1; j <= M; j++)
		{
			B[i][O[j]] = 0;
		}
		pi[i] = 0;
	}
	for (int i = 1; i <= T; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			alpha[i][j] = 0;
			beta[i][j] = 0;
			gamma[i][j] = 0;
		}
	}
}

void calculate_alpha()
{
	for (int i = 1; i <= N; i++)
	{
		alpha[1][i] = pi[i] * B[i][O[1]];
	}
	for (int t = 1; t < T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			long double sum = 0;
			for (int i = 1; i <= N; i++)
			{
				sum += alpha[t][i] * A[i][j];
			}
			alpha[t + 1][j] = sum * B[j][O[t + 1]];
		}
	}

	FILE *fp=fopen("alpha.txt","w");
	for (int t = 1; t <= T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp,"%e\t", alpha[t][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

long double calculate_score()
{
	long double probability = 0;
	for (int i = 1; i <= N; i++)
	{
		probability += alpha[T][i];
	}
	return probability;
}

void calculate_beta()
{
	for (int i = 1; i <= N; i++)
	{
		beta[T][i] = 1;
	}
	for (int t = T - 1; t >= 1; t--)
	{
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				beta[t][i] += A[i][j] * B[j][O[t + 1]] * beta[t + 1][j];
			}
		}
	}
	FILE *fp=fopen("beta.txt","w");
	for (int t = 1; t < T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp,"%e\t", beta[t][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

void predict_state_sequence(){
	for (int t = 1; t <= T; t++)
	{
		long double max = 0;
		int index = 0;
		for (int j = 1; j <= N; j++)
		{
			if (gamma[t][j] > max)
			{
				max = gamma[t][j];
				index = j;
			}
		}
		q[t] = index;
	}
	FILE* fp=fopen("predicted_seq_gamma.txt","w");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",O[t]);
	}
	fprintf(fp,"\n");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",q[t]);
	}
	fprintf(fp,"\n");
	fclose(fp);
}

void calculate_gamma()
{
	for (int t = 1; t <= T; t++)
	{
		long double sum = 0;
		for (int i = 1; i <= N; i++)
		{
			sum += alpha[t][i] * beta[t][i];
		}
		for (int i = 1; i <= N; i++)
		{
			gamma[t][i] = alpha[t][i] * beta[t][i] / sum;
		}
	}
	FILE *fp=fopen("gamma.txt","w");
	for (int t = 1; t <= T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp,"%.16e\t", gamma[t][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	predict_state_sequence();
}

//Solution to Problem2 Of HMM
void viterbi_algo(){
	//Initialization
	for(int i=1;i<=N;i++){
		delta[1][i]=pi[i]*B[i][O[1]];
		psi[1][i]=0;
	}
	for(int t=2;t<=T;t++){
		for(int j=1;j<=N;j++){
			long double max=DBL_MIN;
			int index=0;
			for(int i=1;i<=N;i++){
				if(delta[t-1][i]*A[i][j]>max){
					max=delta[t-1][i]*A[i][j];
					index=i;
				}
			}
			delta[t][j]=max*B[j][O[t]];
			psi[t][j]=index;
		}
	}
	P_star=DBL_MIN;
	for(int i=1;i<=N;i++){
		if(delta[T][i]>P_star){
			P_star=delta[T][i];
			q_star[T]=i;
		}
	}
	for(int t=T-1;t>=1;t--){
		q_star[t]=psi[t+1][q_star[t+1]];
	}
	FILE* fp=fopen("predicted_seq_viterbi.txt","w");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",O[t]);
	}
	fprintf(fp,"\n");
	for(int t=1;t<=T;t++){
		fprintf(fp,"%4d\t",q_star[t]);
	}
	fprintf(fp,"\n");
	fclose(fp);
}

//Calculate XI
void calculate_xi(){
	for(int t=1;t<T;t++){
		long double denominator=0.0;
		for(int i=1;i<=N;i++){
			for(int j=1;j<=N;j++){
				denominator+=(alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[t+1][j]);
			}
		}
		for(int i=1;i<=N;i++){
			for(int j=1;j<=N;j++){
				xi[t][i][j]=(alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[t+1][j])/denominator;
			}
		}
	}
}

void re_estimation(){
	for(int i=1;i<=N;i++){
		pi_bar[i]=gamma[1][i];
	}
	for(int i=1;i<=N;i++){
		int mi=0;
		long double max_value=DBL_MIN;
		long double adjust_sum=0;
		for(int j=1;j<=N;j++){
			long double numerator=0.0, denominator=0.0;
			for(int t=1;t<=T-1;t++){
				numerator+=xi[t][i][j];
				denominator+=gamma[t][i];
			}
			A_bar[i][j]=(numerator/denominator);
			if(A_bar[i][j]>max_value){
				max_value=A_bar[i][j];
				mi=j;
			}
			adjust_sum+=A_bar[i][j];
		}
		A_bar[i][mi]+=(1-adjust_sum);
	}
	for(int j=1;j<=N;j++){
		int mi=0;
		long double max_value=DBL_MIN;
		long double adjust_sum=0;
		for(int k=1;k<=M;k++){
			long double numerator=0.0, denominator=0.0;
			for(int t=1;t<=T;t++){
					if(O[t]==k){
						numerator+=gamma[t][j];
					}
					denominator+=gamma[t][j];
			}
			B_bar[j][k]=(numerator/denominator);
			if(B_bar[j][k]>max_value){
				max_value=B_bar[j][k];
				mi=k;
			}
			if(B_bar[j][k]<1.00e-030){
				B_bar[j][k]=1.00e-030;
			}
			adjust_sum+=B_bar[j][k];
		}
		B_bar[j][mi]+=(1-adjust_sum);
	}
	
	for(int i=1;i<=N;i++){
		pi[i]=pi_bar[i];
	}
	for(int i=1;i<=N;i++){
		for(int j=1;j<=N;j++){
			A[i][j]=A_bar[i][j];
		}
	}
	for(int j=1;j<=N;j++){
		for(int k=1;k<=M;k++){
			B[j][k]=B_bar[j][k];
		}
	}
}

void set_initial_model(){
	for(int d=0;d<=9;d++){
		char srcfnameA[40];
		_snprintf(srcfnameA,40,"initial/A_%d.txt",d);
		char srcfnameB[40];
		_snprintf(srcfnameB,40,"initial/B_%d.txt",d);
		char destfnameA[40];
		_snprintf(destfnameA,40,"initial_model/A_%d.txt",d);
		char destfnameB[40];
		_snprintf(destfnameB,40,"initial_model/B_%d.txt",d);
		char copyA[100];
		_snprintf(copyA,100,"copy /Y %s %s >nul",srcfnameA,destfnameA);
		char copyB[100];
		_snprintf(copyB,100,"copy /Y %s %s >nul",srcfnameB,destfnameB);
		system(copyA);
		system(copyB);
	}
	
}

void initial_model(int d){
	FILE *fp;
	initialization();
	char filenameA[40];
	_snprintf(filenameA,40,"initial_model/A_%d.txt",d);
	fp = fopen(filenameA, "r");
	if (fp == NULL)
	{
		printf("Error\n");
	}
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fscanf(fp, "%Lf ", &A[i][j]);
		}
	}
	fclose(fp);

	char filenameB[40];
	_snprintf(filenameB,40,"initial_model/B_%d.txt",d);
	fp = fopen(filenameB, "r");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fscanf(fp, "%Lf ", &B[i][j]);
		}
	}
	fclose(fp);

	fp = fopen("initial_model/pi.txt", "r");
	for (int i = 1; i <= N; i++)
	{
		fscanf(fp, "%Lf ", &pi[i]);
	}
	fclose(fp);

	fp=fopen("o.txt","r");
	for (int i = 1; i <= T; i++)
	{
		fscanf(fp, "%d\t", &O[i]);
	}
	fclose(fp);
}
void train_model(int digit, int utterance){
	int m=0;
	do{
		calculate_alpha();
		calculate_beta();
		calculate_gamma();
		P_star_dash=P_star;
		viterbi_algo();
		calculate_xi();
		re_estimation();
		m++;
	}while(m<60 && P_star > P_star_dash);
	printf("pstar after iteration : %2d    ------>    %e\n",m,P_star);
	
	FILE *fp;
	char filenameA[40];
	_snprintf(filenameA,40,"234101039_lambda/A_%d_%d.txt",digit,utterance);
	fp=fopen(filenameA,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp, "%e ", A[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);

	char filenameB[40];
	_snprintf(filenameB,40,"234101039_lambda/B_%d_%d.txt",digit,utterance);
	fp=fopen(filenameB,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fprintf(fp, "%e ", B[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

void calculate_avg_model_param(int d){
	long double A_sum[N+1][N+1]={0};
	long double B_sum[N+1][M+1]={0};
	long double temp;
	FILE* fp;
	for(int u=1;u<=25;u++){
		char filenameA[40];
		_snprintf(filenameA,40,"234101039_lambda/A_%d_%d.txt",d,u);
		fp=fopen(filenameA,"r");
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				fscanf(fp, "%Lf ", &temp);
				A_sum[i][j]+=temp;
			}
		}
		fclose(fp);
		char filenameB[40];
		_snprintf(filenameB,40,"234101039_lambda/B_%d_%d.txt",d,u);
		fp=fopen(filenameB,"r");
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= M; j++)
			{
				fscanf(fp, "%Lf ", &temp);
				B_sum[i][j]+=temp;
			}
		}
		fclose(fp);
	}
	FILE* avgfp;
	char fnameA[40];
	_snprintf(fnameA,40,"initial_model/A_%d.txt",d);
	avgfp=fopen(fnameA,"w");
	for(int i=1;i<=N;i++){
		for(int j=1;j<=N;j++){
			A[i][j]=A_sum[i][j]/25;
			fprintf(avgfp,"%e ", A[i][j]);
		}
		fprintf(avgfp,"\n");
	}
	fclose(avgfp);
	char fnameB[40];
	_snprintf(fnameB,40,"initial_model/B_%d.txt",d);
	avgfp=fopen(fnameB,"w");
	for(int i=1;i<=N;i++){
		for(int j=1;j<=M;j++){
			B[i][j]=B_sum[i][j]/25;
			fprintf(avgfp,"%e ", B[i][j]);
		}
		fprintf(avgfp,"\n");
	}
	fclose(avgfp);
}

void store_final_lambda(int digit){
	FILE *fp;
	char filenameA[40];
	_snprintf(filenameA,40,"234101039_lambda/A_%d.txt",digit);
	fp=fopen(filenameA,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp, "%e ", A[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	char filenameB[40];
	_snprintf(filenameB,40,"234101039_lambda/B_%d.txt",digit);
	fp=fopen(filenameB,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fprintf(fp, "%e ", B[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}


void processTestFile(int d){
	FILE *fp;
	initialization();
	char filenameA[40];
	_snprintf(filenameA,40,"234101039_lambda/A_%d.txt",d);
	fp=fopen(filenameA,"r");
	if (fp == NULL){
		printf("Error\n");
	}
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= N; j++){
			fscanf(fp, "%Lf ", &A[i][j]);
		}
	}
	fclose(fp);

	char filenameB[40];
	_snprintf(filenameB,40,"234101039_lambda/B_%d.txt",d);
	fp=fopen(filenameB,"r");
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= M; j++){
			fscanf(fp, "%Lf ", &B[i][j]);
		}
	}
	fclose(fp);

	fp = fopen("initial_model/pi.txt", "r");
	for (int i = 1; i <= N; i++)
	{
		fscanf(fp, "%Lf ", &pi[i]);
	}
	fclose(fp);

	fp=fopen("o.txt","r");
	for (int i = 1; i <= T; i++)
	{
		fscanf(fp, "%d\t", &O[i]);
	}
	fclose(fp);
}

int recognize_digit(){
	int rec_digit=10;
	long double max_prob=DBL_MIN;
	for(int d=0;d<=9;d++){
		processTestFile(d);
		calculate_alpha();
		long double prob=calculate_score();
		if(prob>max_prob){
			max_prob=prob;
			rec_digit=d;
		}
	}
	return rec_digit;
}

void train_HMM(){
	set_initial_model();
	for(int d=0;d<=9;d++){
		for(int t=1;t<=2;t++){
			for(int u=1;u<=TRAIN_SIZE;u++){
				char filename[40];
				_snprintf(filename,40,"234101039_dataset/234101039_E_%d_%d.txt",d,u);
				generate_observation_sequence(filename);
				initial_model(d);
				train_model(d,u);
			}
			calculate_avg_model_param(d);
		}
		store_final_lambda(d);
	}
}

void test_HMM(){
	double accuracy=0.0;
	for(int d=0;d<=9;d++){
		double cnt=0.0;
		printf("------------------------------------ Testing digit %d ------------------------------------\n\n",d);
		for(int u=21;u<=30;u++){
			char filename[40];
			_snprintf(filename,40,"234101039_dataset/234101039_E_%d_%d.txt",d,u);
			generate_observation_sequence(filename);
			int rd=recognize_digit();
			printf("Recognized Digit:%d\n",rd);
			if(rd==d){
				accuracy+=1.0;
				cnt+=1.0;
			}
		}
		printf("\nAccuracy of digit %d : %.1f%c\n\n\n",d,(cnt/10.0)*100,'%');
	}
	printf("Total accuracy of system : %.1f%c\n",accuracy,'%');
}

void process_live_data(char filename[100]){
	FILE *fp;
	char prefixf[100]="live_input/";
	strcat(prefixf,filename);
	fp=fopen(prefixf,"r");
	int samples[13000];
	int x=0;
	for(int i=0;!feof(fp);i++){
		fscanf(fp,"%d",&x);
		if(i>=6000 && i<19000){
			samples[i-6000]=x;
		}
	}
	fclose(fp);
	char prefix[100]="live_input/processed_";
	strcat(prefix,filename);
	fp=fopen(prefix,"w");
	for(int i=0;i<13000;i++){
		fprintf(fp,"%d\n",samples[i]);
	}
	fclose(fp);
}

void live_test_HMM(){
	Sleep(2000);
	system("Recording_Module.exe 2 live_input/test.wav live_input/test.txt");
	generate_observation_sequence("live_input/test.txt");
	int rd=recognize_digit();
	printf("Recognized Digit:%d\n",rd);		
}

int _tmain(int argc, _TCHAR* argv[])
{
	
	printf("Creating codebook ...\n");

	generate_universe();
	generate_codebook();
	load_codebook();

	printf("Codebook created ...\n");

	//Training HMM

	printf("\n\n<---------------------------- Training started ---------------------------->\n\n");

	train_HMM();

	//Testing HMM
	printf("\n\n<-------------------------------------------------------- Testing HMM -------------------------------------------------------->\n\n");
	test_HMM();
	while(1)
		live_test_HMM();

	printf("\n\n<--------------------------------------------------------  ThankYou -------------------------------------------------------->\n\n");
	
	system("pause");

	return 0;
}