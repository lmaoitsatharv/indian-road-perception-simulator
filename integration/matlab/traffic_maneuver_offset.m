function offset=traffic_maneuver_offset(progress,maneuver)
% Continuous position profile with zero lateral slope/curvature at endpoints.
offset=2.5+zeros(size(progress));
if maneuver.mode==0,return;end
q=progress-maneuver.start_progress;
out=q>=0 & q<maneuver.out_length;
pass=q>=maneuver.out_length & q<maneuver.return_start;
back=q>=maneuver.return_start & q<maneuver.total_length;
blend=@(x) 10*x.^3-15*x.^4+6*x.^5;
offset(out)=2.5+(maneuver.pass_offset-2.5)*blend(q(out)/maneuver.out_length);
offset(pass)=maneuver.pass_offset;
offset(back)=maneuver.pass_offset+(2.5-maneuver.pass_offset)*blend((q(back)-maneuver.return_start)/maneuver.out_length);
end
