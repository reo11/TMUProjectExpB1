%% knnƒ‚ƒfƒ‹‚Ìì¬

[fDB, C] = labeling(DB); %“Á’¥—Ê‚Ìƒ‰ƒxƒŠƒ“ƒO
model = fitcknn(fDB, C);
model.NumNeighbors = 5; 