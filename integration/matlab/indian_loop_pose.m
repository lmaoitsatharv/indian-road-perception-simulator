function [xy, heading, curvature, loopLength] = indian_loop_pose(s, offset)
%INDIAN_LOOP_POSE Synthetic rounded-block road, fixed world metres/radians.
% Counterclockwise road reference. Positive offset is its left (inner) side;
% clockwise traffic uses negative offset, also left relative to its heading.
if nargin<2, offset=0; end
s=s(:); if isscalar(offset), offset=repmat(offset,size(s)); else, offset=offset(:); end
r=18; lengths=[84,pi*r/2,44,pi*r/2,84,pi*r/2,44,pi*r/2];
edges=[0,cumsum(lengths)]; loopLength=edges(end); s=mod(s,loopLength);
xy=zeros(numel(s),2);heading=zeros(numel(s),1);curvature=zeros(numel(s),1);
for part=1:8
    use=s>=edges(part) & s<edges(part+1); q=s(use)-edges(part);
    switch part
        case 1, xy(use,:)=[38+q,20+zeros(size(q))];heading(use)=0;
        case 2, a=-pi/2+q/r;xy(use,:)=[122+r*cos(a),38+r*sin(a)];heading(use)=a+pi/2;
        case 3, xy(use,:)=[140+zeros(size(q)),38+q];heading(use)=pi/2;
        case 4, a=q/r;xy(use,:)=[122+r*cos(a),82+r*sin(a)];heading(use)=a+pi/2;
        case 5, xy(use,:)=[122-q,100+zeros(size(q))];heading(use)=pi;
        case 6, a=pi/2+q/r;xy(use,:)=[38+r*cos(a),82+r*sin(a)];heading(use)=a+pi/2;
        case 7, xy(use,:)=[20+zeros(size(q)),82-q];heading(use)=-pi/2;
        case 8, a=pi+q/r;xy(use,:)=[38+r*cos(a),38+r*sin(a)];heading(use)=a+pi/2;
    end
    if mod(part,2)==0,curvature(use)=1/r;end
end
xy=xy+offset.*[-sin(heading),cos(heading)];
heading=atan2(sin(heading),cos(heading));
end
