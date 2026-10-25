function fig=replay_indian_traffic_3d(runDir,options)
%REPLAY_INDIAN_TRAFFIC_3D Native MATLAB 3D view of saved simulation states.
% Camera and playback changes never change the recorded experiment.
arguments
    runDir (1,:) char
    options.AutoPlay (1,1) logical = false
    options.Camera (1,1) string = "Follow car"
end
assert(ismember(options.Camera,["Follow car","Driver view","Overview"]), ...
    'Tesla:ReplayCamera','Camera must be Follow car, Driver view or Overview.');
data=load_traffic_replay(runDir);
cfg=data.config;specs=data.specs;ego=data.egoLog;actors=data.actorLog;
duration=ego(end,1);
fig=figure('Name','Tesla | Indian traffic in 3D','NumberTitle','off', ...
    'Color',[.055,.075,.10],'Position',[50,60,1380,820],'Tag','TeslaTraffic3D');
ax=axes(fig,'Position',[.015,.20,.775,.75],'Color',[.65,.81,.92]);
hold(ax,'on');axis(ax,'equal');axis(ax,'off');axis(ax,'vis3d');
xlim(ax,[-20,180]);ylim(ax,[-20,140]);zlim(ax,[0,45]);
set(ax,'Projection','perspective','Clipping','off');
roadWorld(ax,cfg.loop_length_m);
egoSpec=struct('kind','car','length_m',4.5,'width_m',1.8,'rear_overhang_m',.9, ...
    'height_m',1.5,'color',[.05,.75,.65]);
egoMesh=makeActor(ax,egoSpec);meshes=cell(numel(specs),1);
for j=1:numel(specs),meshes{j}=makeActor(ax,specs(j));end
light(ax,'Position',[60,-40,120],'Style','local','Color',[1,.96,.87]);
lighting(ax,'gouraud');material(ax,'dull');
panel=uipanel(fig,'Position',[.805,.20,.18,.75],'BackgroundColor',[.09,.12,.16], ...
    'ForegroundColor',[.95,.96,.98],'Title','DRIVING STATE','FontSize',12);
modeText=uicontrol(panel,'Style','text','Units','normalized','Position',[.05,.80,.9,.14], ...
    'FontSize',21,'FontWeight','bold','BackgroundColor',panel.BackgroundColor);
hud=uicontrol(panel,'Style','text','Units','normalized','Position',[.06,.28,.9,.52], ...
    'FontSize',12,'HorizontalAlignment','left','ForegroundColor',[.90,.94,.98], ...
    'BackgroundColor',panel.BackgroundColor);
uicontrol(panel,'Style','text','Units','normalized','Position',[.06,.035,.88,.24], ...
    'String',sprintf('Yellow: auto-rickshaw\nPurple: two-wheeler\nBlue: traffic car\nTeal: your car\n\nSynthetic development world\nRecorded simulation; rule policy'), ...
    'FontSize',10,'HorizontalAlignment','left','ForegroundColor',[.69,.77,.84], ...
    'BackgroundColor',panel.BackgroundColor);
titleText=uicontrol(fig,'Style','text','Units','normalized','Position',[.025,.95,.76,.04], ...
    'String','INDIAN TRAFFIC  /  3D','FontWeight','bold','FontSize',15, ...
    'HorizontalAlignment','left','ForegroundColor',[.96,.98,1],'BackgroundColor',fig.Color);
playButton=uicontrol(fig,'Style','pushbutton','String','Play','Tag','Play', ...
    'Units','normalized','Position',[.025,.105,.085,.05],'Callback',@togglePlay);
uicontrol(fig,'Style','pushbutton','String','Reset','Tag','Reset', ...
    'Units','normalized','Position',[.12,.105,.085,.05],'Callback',@resetReplay);
cameraControl=uicontrol(fig,'Style','popupmenu','String',{'Follow car','Driver view','Overview'}, ...
    'Tag','Camera','Units','normalized','Position',[.225,.105,.15,.05],'Callback',@changeCamera);
cameraChoice=find(strcmp(cameraControl.String,options.Camera),1);
if ~isempty(cameraChoice),cameraControl.Value=cameraChoice;end
rateControl=uicontrol(fig,'Style','popupmenu','String',{'1x','4x','16x'},'Tag','Rate', ...
    'Units','normalized','Position',[.395,.105,.08,.05],'Callback',@changeRate);
repeat=uicontrol(fig,'Style','checkbox','String','Repeat','Value',1,'Tag','Repeat', ...
    'Units','normalized','Position',[.50,.11,.10,.04],'ForegroundColor',[.9,.93,.97],'BackgroundColor',fig.Color);
seek=uicontrol(fig,'Style','slider','Min',0,'Max',duration,'Value',0,'Tag','Seek', ...
    'SliderStep',min(1,[.1,10]/duration), ...
    'Units','normalized','Position',[.025,.06,.75,.025],'Callback',@seekReplay);
revision='saved rollout';if isfield(cfg,'revision'),revision=cfg.revision;end
uicontrol(fig,'Style','text','Units','normalized','Position',[.025,.004,.95,.038], ...
    'String',sprintf('Seed %d  |  %d moving vehicles  |  %d pedestrians  |  %s  |  saved run: %s', ...
    cfg.seed,cfg.vehicle_count,cfg.pedestrian_count,revision,string(lastFolder(runDir))), ...
    'HorizontalAlignment','left','FontSize',10,'ForegroundColor',[.67,.75,.83],'BackgroundColor',fig.Color);
playing=options.AutoPlay;simTime=0;baseTime=0;rate=1;wall=tic;
if playing,playButton.String='Pause';end
clockTimer=timer('ExecutionMode','fixedSpacing','Period',.05,'BusyMode','drop', ...
    'Name','TeslaTraffic3DReplay','TimerFcn',@tick,'ErrorFcn',@timerError);
fig.CloseRequestFcn=@closeReplay;
fig.DeleteFcn=@deleteReplay;
setappdata(fig,'ReplayTimer',clockTimer);setappdata(fig,'RunDirectory',runDir);
setappdata(fig,'MainAxes',ax);setappdata(fig,'EgoTransform',egoMesh.root);
setappdata(fig,'IsPlaying',playing);setappdata(fig,'PlaybackRate',rate);
render();if playing,start(clockTimer);end

    function tick(~,~)
        if ~isgraphics(fig),return;end
        if ~playing,return;end
        advanceTime();
        render();
    end
    function advanceTime()
        if ~playing,return;end
        simTime=baseTime+toc(wall)*rate;
        if simTime>=duration
            if repeat.Value,simTime=mod(simTime,duration);baseTime=simTime;wall=tic;
            else,simTime=duration;setPlaying(false);end
        end
    end
    function setPlaying(value)
        playing=value;baseTime=simTime;wall=tic;
        setappdata(fig,'IsPlaying',playing);
        if playing
            playButton.String='Pause';
            if strcmp(clockTimer.Running,'off'),start(clockTimer);end
        else
            playButton.String='Play';
            if strcmp(clockTimer.Running,'on'),stop(clockTimer);end
        end
    end
    function togglePlay(~,~)
        wasPlaying=playing;advanceTime();
        if ~wasPlaying && simTime>=duration,simTime=0;end
        setPlaying(~wasPlaying);render();
    end
    function resetReplay(~,~)
        simTime=0;setPlaying(false);render();
    end
    function changeRate(~,~)
        advanceTime();choices=[1,4,16];rate=choices(rateControl.Value);
        baseTime=simTime;wall=tic;setappdata(fig,'PlaybackRate',rate);render();
    end
    function changeCamera(~,~),render();end
    function seekReplay(~,~)
        simTime=max(0,min(duration,seek.Value));baseTime=simTime;wall=tic;render();
    end
    function render()
        k=find(ego(:,1)<=simTime,1,'last');if isempty(k),k=1;end
        next=min(k+1,size(ego,1));a=0;
        if next>k,a=(simTime-ego(k,1))/(ego(next,1)-ego(k,1));end
        p=interpolatePose(ego(k,2:4),ego(next,2:4),a);
        setPose(egoMesh,p,0,0);
        state=reshape(actors(k,:,:),numel(specs),size(actors,3));
        future=reshape(actors(next,:,:),numel(specs),size(actors,3));
        for j=1:numel(specs)
            ap=interpolatePose(state(j,1:3),future(j,1:3),a);
            speed=norm(state(j,4:5));setPose(meshes{j},ap,simTime,speed);
        end
        u=[cos(p(3)),sin(p(3))];n=[-u(2),u(1)];
        if cameraControl.Value==1
            cameraPosition=[p(1:2)-10*u+1.4*n,5.8];cameraTarget=[p(1:2)+7*u,1.0];angle=64;
            egoMesh.root.Visible='on';
        elseif cameraControl.Value==2
            cameraPosition=[p(1:2)+1.35*u-.35*n,1.35];cameraTarget=[p(1:2)+26*u-.35*n,1.25];angle=78;
            egoMesh.root.Visible='off';
        else
            cameraPosition=[80,-100,145];cameraTarget=[80,60,0];angle=53;egoMesh.root.Visible='on';
        end
        campos(ax,cameraPosition);camtarget(ax,cameraTarget);camup(ax,[0,0,1]);camva(ax,angle);
        setappdata(fig,'DisplayedEgoPose',p);setappdata(fig,'DisplayedTime',simTime);
        words={'PROCEED','SLOW','STOP'};colors=[.12,.85,.69;1,.73,.18;1,.35,.30];mode=round(ego(k,10))+1;
        modeText.String=words{mode};modeText.ForegroundColor=colors(mode,:);
        braking='off';if ego(k,11),braking='ACTIVE';end
        maneuver='Follow lane';
        if size(ego,2)>=13
            labels={'Follow lane','Move out','Overtake','Return to lane'};maneuver=labels{min(4,max(1,round(ego(k,13))+1))};
        end
        hud.String=sprintf('Speed       %.1f km/h\nTarget      %.1f km/h\nTime        %.1f / %.0f s\nPlayback    %dx\n\n%s\nBraking: %s\nCrossing now: %d\nLaps: %d\n\nRun ego collisions: %d\nOther overlap samples: %d', ...
            ego(k,5)*3.6,ego(k,8)*3.6,simTime,ego(end,1),rate,maneuver,braking, ...
            sum(state(cfg.vehicle_count+(1:cfg.pedestrian_count),6)==1),floor(ego(k,9)/cfg.loop_length_m), ...
            data.result.ego_collision_samples,data.result.non_ego_overlap_samples_at_0_5s);
        titleText.String=sprintf('INDIAN TRAFFIC  /  %s',upper(cameraControl.String{cameraControl.Value}));
        seek.Value=max(0,min(duration,simTime));drawnow limitrate nocallbacks;
    end
    function closeReplay(~,~)
        delete(fig);
    end
    function deleteReplay(~,~)
        playing=false;
        if isvalid(clockTimer),stop(clockTimer);delete(clockTimer);end
    end
    function timerError(~,event)
        if ~isgraphics(fig),return;end
        setappdata(fig,'LastReplayError',event.Data);
        setPlaying(false);
        titleText.String='Replay paused after a rendering error; see LastReplayError in figure appdata.';
    end
end

function value=lastFolder(path)
[~,value]=fileparts(path);
end
function p=interpolatePose(a,b,f)
p=a;p(1:2)=a(1:2)+f*(b(1:2)-a(1:2));
p(3)=a(3)+f*atan2(sin(b(3)-a(3)),cos(b(3)-a(3)));
end
function setPose(actor,p,time,speed)
actor.root.Matrix=makehgtform('translate',[p(1:2),0],'zrotate',p(3));
if ~isempty(actor.legs)
    phase=sin(time*min(13,5+4*speed))*min(.55,speed*.5);
    actor.legs(1).Matrix=makehgtform('translate',[0,-.13,.82],'yrotate',phase);
    actor.legs(2).Matrix=makehgtform('translate',[0,.13,.82],'yrotate',-phase);
    actor.arms(1).Matrix=makehgtform('translate',[0,-.27,1.36],'yrotate',-phase);
    actor.arms(2).Matrix=makehgtform('translate',[0,.27,1.36],'yrotate',phase);
end
end
function actor=makeActor(ax,spec)
root=hgtransform('Parent',ax);actor=struct('root',root,'legs',[],'arms',[]);
if strcmp(spec.kind,'pedestrian')
    actor.legs=gobjects(2,1);actor.arms=gobjects(2,1);
    box(root,[-.13,-.23,.81],[.26,.46,.62],spec.color);
    box(root,[-.12,-.12,1.45],[.24,.24,.26],[.69,.45,.30]);
    for k=1:2
        actor.legs(k)=hgtransform('Parent',root);
        box(actor.legs(k),[-.075,-.075,-.76],[.15,.15,.76],[.15,.19,.29]);
        actor.arms(k)=hgtransform('Parent',root);
        box(actor.arms(k),[-.07,-.07,-.54],[.14,.14,.54],spec.color);
    end
    return
end
L=spec.length_m;W=spec.width_m;rear=spec.rear_overhang_m;
if strcmp(spec.kind,'two_wheeler')
    box(root,[-rear,-W*.25,.38],[L,.5*W,.3],spec.color);
    box(root,[.2,-.2,.69],[.50,.4,.22],[.06,.07,.09]);
    box(root,[.35,-.20,.92],[.30,.4,.52],spec.color);
    box(root,[.39,-.13,1.45],[.24,.26,.23],[.85,.86,.88]);
    for x=[-.10,L-rear-.15],box(root,[x-.20,-.08,.03],[.40,.16,.45],[.045,.05,.065]);end
elseif strcmp(spec.kind,'auto_rickshaw')
    box(root,[-rear,-W/2,.36],[L,W,.50],[.12,.37,.18]);
    box(root,[-rear,-W/2,1.40],[L*.81,W,.22],spec.color);
    box(root,[-rear,-W/2,.8],[L*.18,W,.60],spec.color);
    box(root,[L*.61-rear,-W/2,.85],[.10,W,.52],[.22,.37,.43]);
    for y=[-W*.47,W*.47],box(root,[-.23,y-.10,.07],[.5,.20,.55],[.04,.045,.05]);end
    box(root,[L-rear-.43,-.10,.07],[.43,.20,.5],[.04,.045,.05]);
else
    box(root,[-rear,-W/2,.30],[L,W,.60],spec.color);
    box(root,[L*.18-rear,-W*.43,.90],[L*.52,W*.86,.54],[.14,.24,.30]);
    box(root,[L*.18-rear,-W*.45,1.42],[L*.52,W*.9,.09],spec.color);
    for x=[0,L*.66]
        for y=[-W*.5,W*.5],box(root,[x-.30,y-.11,.05],[.60,.22,.59],[.035,.04,.05]);end
    end
    for y=[-W*.35,W*.25]
        box(root,[L-rear+.005,y,.60],[.025,W*.13,.13],[1,.96,.67]);
        box(root,[-rear-.03,y,.60],[.025,W*.13,.13],[.92,.12,.10]);
    end
end
end
function h=box(parent,origin,dims,color)
v=[0,0,0;1,0,0;1,1,0;0,1,0;0,0,1;1,0,1;1,1,1;0,1,1].*dims+origin;
f=[1,4,3,2;1,2,6,5;2,3,7,6;3,4,8,7;4,1,5,8;5,6,7,8];
h=patch('Parent',parent,'Vertices',v,'Faces',f,'FaceColor',color, ...
    'EdgeColor','none','FaceLighting','gouraud','AmbientStrength',.65,'DiffuseStrength',.6);
end
function roadWorld(ax,L)
[skyX,skyY,skyZ]=sphere(24);
surf(ax,500*skyX+80,500*skyY+60,500*skyZ,'FaceColor',[.64,.81,.93], ...
    'EdgeColor','none','FaceLighting','none');
box(ax,[-30,-30,-.25],[220,180,.20],[.31,.44,.29]);
s=linspace(0,L,1000)';
strip(ax,s,-8,-5.5,.14,[.62,.60,.55]);strip(ax,s,5.5,8,.14,[.62,.60,.55]);
strip(ax,s,-5.5,5.5,.025,[.22,.24,.26]);
for side=[-1,1]
    for k=1:3:numel(s)-3
        color=[.12,.13,.13];if mod(floor(k/3),2)==0,color=[.90,.71,.17];end
        strip(ax,s(k:k+3),side*5.5,side*5.65,.18,color);
    end
    edge=indian_loop_pose(s,side*5.05);plot3(ax,edge(:,1),edge(:,2),.04*ones(size(s)),'Color',[.90,.90,.85],'LineWidth',1.1);
end
for k=0:7:L-4
    if k>=15 && k<=65,continue;end
    strip(ax,(k:.4:k+3)',-.07,.07,.04,[.86,.77,.45]);
end
buildings=[47,43,16,13,12;69,44,16,13,18;91,42,20,15,10;48,66,21,13,15;80,69,30,11,12];
colors=[.73,.57,.41;.76,.71,.58;.45,.58,.61;.79,.72,.63;.66,.59,.50];
for i=1:size(buildings,1)
    b=buildings(i,:);box(ax,[b(1:2),0],b(3:5),colors(i,:));
    for x=b(1)+1.5:3:b(1)+b(3)-1.5
        for z=2:3:b(5)-1,box(ax,[x,b(2)-.02,z],[1.2,.04,1.5],[.18,.30,.35]);end
    end
end
% Low-poly trees and lamps are decoration; collision map remains explicit.
for k=8:24:L
    [p,h]=indian_loop_pose(k,10.5);box(ax,[p-.14,0],[.28,.28,2.6],[.37,.27,.16]);
    [x,y,z]=sphere(7);surf(ax,1.5*x+p(1),1.5*y+p(2),1.8*z+3.1, ...
        'FaceColor',[.14,.38,.17],'EdgeColor','none');
    if mod(round(k),48)==8
        [p,h]=indian_loop_pose(k,-7.5);box(ax,[p-.06,0],[.12,.12,5],[.37,.41,.43]);
        box(ax,[p-.3,4.9],[.6,.35,.15],[.87,.87,.74]);
    end
end
end
function strip(ax,s,a,b,z,color)
p=indian_loop_pose(s,a);q=indian_loop_pose(s,b);n=numel(s);
vertices=[p,z*ones(n,1);q,z*ones(n,1)];k=(1:n-1)';faces=[k,k+1,k+1+n,k+n];
patch(ax,'Vertices',vertices,'Faces',faces,'FaceColor',color,'EdgeColor','none','AmbientStrength',.8);
end
