function [fDB,C]=labeling(DB,hanbetu)
    %“Á’¥—Ê‚ÌDBì¬ŠÖ”

   if(hanbetu==5)
             DB_filename='M:\project\dataset4\DB\csv\FP.csv';
             feature_list = csvread(DB_filename); 
   end

    for i=1:200
         %DB‚Ìi–‡–Ú‚ğ“Ç‚İ‚Ş
         img = DB(:,:,i);
         
         %i–‡–Ú‚Ì“Á’¥—ÊŒvZ
         if(hanbetu == 3)
             feature = funcDCT(img); %dct‚Ìê‡
         elseif(hanbetu == 4)
             feature = funcHIST(img); %hist‚Ìê‡
         elseif(hanbetu == 5)
             feature = feature_list(i,:);
         elseif(hanbetu == 6)
             feature = funcHOG(img);
         end
         
         %“Á’¥—Ê‚ÌDB
         fDB(:,i) = feature;
         
         %³‰ğƒ‰ƒxƒ‹
         C(i) = fix((i-1)/10);
    end
    
    %s‚Æ—ñ‚Ì“ü‚ê‘Ö‚¦
    C = transpose(C);
    fDB = transpose(fDB);
end

