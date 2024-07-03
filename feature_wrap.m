function feature_2d=feature_wrap(feature_3d)
    feature_2d=zeros(size(feature_3d,2),310);
    for i=1:size(feature_3d,2)
         feature_2d(i,:)=reshape(feature_3d(:,i,:),1,310);
    end
end
         
