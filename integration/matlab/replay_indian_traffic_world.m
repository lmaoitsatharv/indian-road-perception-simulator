function fig=replay_indian_traffic_world(runDir,options)
%REPLAY_INDIAN_TRAFFIC_WORLD Interactive playback of an actual saved rollout.
% Play/Pause, seek and speed change affect playback only, not the experiment.
arguments
    runDir (1,:) char
    options.AutoPlay (1,1) logical = false
end
data=load_traffic_replay(runDir);
cfg=data.config;specs=data.specs;ego=data.egoLog;actors=data.actorLog;
duration=ego(end,1);
fig=figure('Name','Indian traffic world - cars, autos, bikes and pedestrians', ...
    'Color',[.07,.09,.12],'NumberTitle','off','Position',[60,70,1280,780],'Tag','TeslaTraffic2D');
ax=axes(fig,'Position',[.035,.22,.73,.70],'Color',[.13,.20,.17], ...
    'XColor',[.75,.8,.84],'YColor',[.75,.8,.84],'FontSize',10);
hold(ax,'on');axis(ax,'equal');xlim(ax,[0,160]);ylim(ax,[0,120]);
xlabel(ax,'World X (metres)');ylabel(ax,'World Y (metres)');
s=(0:.4:cfg.loop_length_m)';
outer=indian_loop_pose(s,-8);outerRoad=indian_loop_pose(s,-5.5);
innerRoad=indian_loop_pose(s,5.5);inner=indian_loop_pose(s,8);
ring=subtract(polyshape(outer(:,1),outer(:,2)),polyshape(inner(:,1),inner(:,2)));
plot(ax,ring,'FaceColor',[.69,.64,.52],'FaceAlpha',1,'EdgeColor','none');
road=subtract(polyshape(outerRoad(:,1),outerRoad(:,2)),polyshape(innerRoad(:,1),innerRoad(:,2)));
plot(ax,road,'FaceColor',[.22,.24,.27],'FaceAlpha',1,'EdgeColor',[.85,.82,.69]);
center=indian_loop_pose(s,0);plot(ax,center(:,1),center(:,2),'--','Color',[.78,.73,.46],'LineWidth',.8);
% A short unmarked section keeps markings from defining the whole road.
unmarked=s>=15 & s<=65;plot(ax,center(unmarked,1),center(unmarked,2),'Color',[.22,.24,.27],'LineWidth',2);
plot(ax,data.route(:,2),data.route(:,3),':','Color',[.25,.71,.83],'LineWidth',1);
buildings=[50,42,15,13;72,42,18,13;97,42,12,13;50,66,19,13;80,68,28,12];
labels={'Shops','Market','Homes','Bus stop area','Homes'};
for i=1:size(buildings,1)
    b=buildings(i,:);rectangle(ax,'Position',b,'FaceColor',[.36,.36,.39],'EdgeColor',[.48,.48,.5]);
    text(ax,b(1)+b(3)/2,b(2)+b(4)/2,labels{i},'HorizontalAlignment','center','Color',[.92,.9,.84],'FontSize',8);
end
text(ax,80,60,'MIXED TRAFFIC TEST BLOCK','HorizontalAlignment','center','Color',[.57,.76,.61],'FontSize',9);
text(ax,78,9,'Footpath: walking pedestrians can start crossing anywhere on straight sections', ...
    'HorizontalAlignment','center','Color',[.9,.86,.73],'FontSize',9);
egoPatch=patch(ax,nan(4,1),nan(4,1),[.13,.82,.70],'EdgeColor','w','LineWidth',1.2);
egoArrow=plot(ax,nan,nan,'w-','LineWidth',2);egoTrail=plot(ax,nan,nan,'Color',[.12,.9,.73],'LineWidth',1.5);
handles=gobjects(numel(specs),1);names=gobjects(numel(specs),1);
for i=1:numel(specs)
    if strcmp(specs(i).kind,'pedestrian')
        handles(i)=plot(ax,nan,nan,'o','MarkerSize',5,'MarkerFaceColor',[.98,.55,.3],'MarkerEdgeColor',[1,.9,.7]);
        name=['P',num2str(i-cfg.vehicle_count)];
    else
        handles(i)=patch(ax,nan(4,1),nan(4,1),specs(i).color,'EdgeColor',[.92,.94,.98],'LineWidth',.5);
        if strcmp(specs(i).kind,'auto_rickshaw'),name='AUTO';elseif strcmp(specs(i).kind,'two_wheeler'),name='2W';else,name='';end
    end
    names(i)=text(ax,nan,nan,name,'HorizontalAlignment','center','Color',[.94,.95,.99],'FontSize',7,'Clipping','on');
end
titleHandle=title(ax,'','Color',[.96,.98,1],'FontSize',13,'FontWeight','bold');
panel=uipanel(fig,'Position',[.80,.22,.18,.70],'BackgroundColor',[.11,.14,.18], ...
    'ForegroundColor',[.93,.95,.98],'Title','LIVE STATE','FontSize',12);
status=uicontrol(panel,'Style','text','Units','normalized','Position',[.06,.75,.88,.19], ...
    'BackgroundColor',[.11,.14,.18],'ForegroundColor',[.13,.85,.69],'FontSize',19,'FontWeight','bold');
hud=uicontrol(panel,'Style','text','Units','normalized','Position',[.06,.35,.88,.40], ...
    'BackgroundColor',[.11,.14,.18],'ForegroundColor',[.93,.95,.98],'FontSize',12,'HorizontalAlignment','left');
uicontrol(panel,'Style','text','Units','normalized','Position',[.06,.02,.88,.30], ...
    'BackgroundColor',[.11,.14,.18],'ForegroundColor',[.74,.79,.86],'FontSize',10,'HorizontalAlignment','left', ...
    'String',sprintf('Yellow: auto-rickshaws\nPurple: two-wheelers\nBlue: other cars\nOrange: crossing pedestrian\nPurple dot: footpath walker\n\nGround-truth simulation\nRule baseline; RL pending\nFixed loop; no camera map'));
playButton=uicontrol(fig,'Style','pushbutton','String','Play','Tag','Play','Units','normalized','Position',[.06,.105,.10,.055],'Callback',@togglePlay);
uicontrol(fig,'Style','pushbutton','String','Reset','Tag','Reset','Units','normalized','Position',[.175,.105,.10,.055],'Callback',@resetReplay);
rateControl=uicontrol(fig,'Style','popupmenu','String',{'1x','4x','16x'},'Tag','Rate','Units','normalized', ...
    'Position',[.30,.105,.085,.055],'Callback',@changeRate);
repeat=uicontrol(fig,'Style','checkbox','String','Repeat replay','Value',1,'Tag','Repeat','Units','normalized', ...
    'Position',[.41,.105,.15,.05],'BackgroundColor',fig.Color,'ForegroundColor',[.9,.93,.97]);
seek=uicontrol(fig,'Style','slider','Min',0,'Max',duration,'Value',0,'Tag','Seek','Units','normalized', ...
    'Position',[.06,.06,.70,.025],'SliderStep',min(1,[.1,10]/duration),'Callback',@seekReplay);
parkedCount=sum(strcmp({specs.kind},'parked_car'));
uicontrol(fig,'Style','text','Units','normalized','Position',[.06,.005,.92,.045], ...
    'String',sprintf('Recorded simulation | seed %d | %d traffic vehicles + %d walkers + %d parked vehicles | Same seed = same replay',cfg.seed,cfg.vehicle_count,cfg.pedestrian_count,parkedCount), ...
    'BackgroundColor',fig.Color,'ForegroundColor',[.69,.76,.84],'FontSize',10);
playing=options.AutoPlay;simTime=0;rate=1;baseTime=0;wall=tic;
if playing,playButton.String='Pause';end
clockTimer=timer('ExecutionMode','fixedSpacing','Period',.04,'BusyMode','drop', ...
    'Name','TeslaTraffic2DReplay','TimerFcn',@tick,'ErrorFcn',@timerError);
fig.CloseRequestFcn=@closeReplay;setappdata(fig,'ReplayTimer',clockTimer);setappdata(fig,'RunDirectory',runDir);
fig.DeleteFcn=@deleteReplay;
setappdata(fig,'MainAxes',ax);setappdata(fig,'IsPlaying',playing);setappdata(fig,'PlaybackRate',rate);
render();if playing,start(clockTimer);end

    function tick(~,~)
        if ~isgraphics(fig),return;end
        if playing
            advanceTime();
            render();
        end
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
        playing=value;baseTime=simTime;wall=tic;setappdata(fig,'IsPlaying',playing);
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
    function seekReplay(~,~)
        simTime=max(0,min(duration,seek.Value));baseTime=simTime;wall=tic;render();
    end
    function render()
        k=find(ego(:,1)<=simTime,1,'last');if isempty(k),k=1;end
        next=min(k+1,size(ego,1));fraction=0;
        if next>k,fraction=(simTime-ego(k,1))/(ego(next,1)-ego(k,1));end
        p=ego(k,2:4);p(1:2)=p(1:2)+fraction*(ego(next,2:3)-p(1:2));
        p(3)=p(3)+fraction*atan2(sin(ego(next,4)-p(3)),cos(ego(next,4)-p(3)));
        body=vertices(p,4.5,1.8,.9);set(egoPatch,'XData',body(:,1),'YData',body(:,2));
        nose=p(1:2)+3.5*[cos(p(3)),sin(p(3))];set(egoArrow,'XData',[p(1),nose(1)],'YData',[p(2),nose(2)]);
        trailStart=max(1,k-180);set(egoTrail,'XData',ego(trailStart:k,2),'YData',ego(trailStart:k,3));
        state=reshape(actors(k,:,:),numel(specs),size(actors,3));
        future=reshape(actors(next,:,:),numel(specs),size(actors,3));
        state(:,1:2)=state(:,1:2)+fraction*(future(:,1:2)-state(:,1:2));
        state(:,3)=state(:,3)+fraction*atan2(sin(future(:,3)-state(:,3)),cos(future(:,3)-state(:,3)));
        for j=1:numel(specs)
            if strcmp(specs(j).kind,'pedestrian')
                color=[.66,.48,.91];if state(j,6)==1,color=[1,.47,.19];end
                set(handles(j),'XData',state(j,1),'YData',state(j,2),'MarkerFaceColor',color);
            else
                b=vertices(state(j,1:3),specs(j).length_m,specs(j).width_m,specs(j).rear_overhang_m);
                set(handles(j),'XData',b(:,1),'YData',b(:,2));
            end
            set(names(j),'Position',[state(j,1),state(j,2)+1.4,0]);
        end
        words={'PROCEED','SLOW','STOP'};colors=[.13,.85,.69;1,.72,.18;1,.35,.3];mode=round(ego(k,10))+1;
        status.String=words{mode};status.ForegroundColor=colors(mode,:);
        braking='off';if ego(k,11),braking='ACTIVE';end
        crossings=sum(state(cfg.vehicle_count+(1:cfg.pedestrian_count),6)==1);
        hud.String=sprintf('Speed: %.1f km/h\nTarget: %.1f km/h\nTime: %.1f / %.0f s\nPlayback: %dx\nLaps: %d\nCrossing now: %d\nBraking: %s\nEgo collisions: %d\nOther overlap samples: %d', ...
            ego(k,5)*3.6,ego(k,8)*3.6,simTime,ego(end,1),rate,floor(ego(k,9)/cfg.loop_length_m),crossings,braking, ...
            data.result.ego_collision_samples,data.result.non_ego_overlap_samples_at_0_5s);
        titleHandle.String=sprintf('Indian traffic prototype | left-side driving | seed %d',cfg.seed);
        setappdata(fig,'DisplayedEgoPose',p);setappdata(fig,'DisplayedTime',simTime);
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
        setappdata(fig,'LastReplayError',event.Data);setPlaying(false);
        titleHandle.String='Replay paused after a rendering error; see LastReplayError in figure appdata.';
    end
end

function body=vertices(p,len,width,rear)
body=[-rear,-width/2;len-rear,-width/2;len-rear,width/2;-rear,width/2];
body=body*[cos(p(3)),sin(p(3));-sin(p(3)),cos(p(3))]+p(1:2);
end
