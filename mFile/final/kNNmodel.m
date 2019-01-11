% DB: DB‚Ì“Á’¥—Ê
% X: DB‚Ì³‰ğƒ‰ƒxƒ‹

function Mdl = kNNmodel(feature, label)
    
    Mdl = fitcknn(feature, label, 'NumNeighbors',5,'Standardize',1);
    
end