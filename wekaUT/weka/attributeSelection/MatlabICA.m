DATA = load('ICAdataMatrix.txt');
[IC, A, invA] = fastica(DATA','displayMode','off','approach','symm','g','tanh','stabilization','on');
[ICnumRows, ICnumCols] = size(IC);
[AnumRows, AnumCols] = size(A);
[invAnumRows, invAnumCols] = size(invA);

f1 = fopen('ICAmixingMatrix.txt','w');
% writes out matrix columnwise
fprintf(f1,'%d\n', AnumRows);
fprintf(f1,'%d\n', AnumCols);
fprintf(f1,'%f ', A);
fclose(f1);
f2 = fopen('ICAinverseMixingMatrix.txt','w');
% writes out matrix columnwise
fprintf(f2,'%d\n', invAnumRows);
fprintf(f2,'%d\n', invAnumCols);
fprintf(f2,'%f ', invA);
fclose(f2);
f3 = fopen('ICAindependantComponents.txt','w');
% writes out matrix columnwise
fprintf(f3,'%d\n', ICnumRows);
fprintf(f3,'%d\n', ICnumCols);
fprintf(f3,'%f ', IC);
fclose(f3);
