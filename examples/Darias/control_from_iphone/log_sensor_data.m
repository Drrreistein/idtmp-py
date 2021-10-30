
if ~exist('m', 'var')
    m = mobiledev;
end

while true
    to_save = [m.Acceleration, m.AngularVelocity, m.Orientation];
    disp(to_save)
    %ang_vel = m.AngularVelocity;
    %rot = m.Orientation;
    save ('sensor_data.txt', 'to_save', '-ascii')
end