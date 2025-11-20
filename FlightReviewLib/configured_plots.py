""" This contains the list of all drawn plots on the log plotting page """

from html import escape

from bokeh.layouts import column
from bokeh.models import Range1d
from bokeh.models.widgets import Button
from bokeh.io import curdoc

from scipy import integrate, interpolate, signal

from config import *
from helper import *
from leaflet import ulog_to_polyline
from plotting import *
from plotted_tables import (
    get_logged_messages, get_changed_parameters,
    get_info_table_html, get_heading_html, get_error_labels_html,
    get_hardfault_html, get_corrupt_log_html
    )

from vtol_tailsitter import *

#pylint: disable=cell-var-from-loop, undefined-loop-variable,
#pylint: disable=consider-using-enumerate,too-many-statements



def generate_plots(ulog, px4_ulog, db_data, vehicle_data, link_to_3d_page,
                   link_to_pid_analysis_page):
    """ create a list of bokeh plots (and widgets) to show """

    plots = []
    data = ulog.data_list

    # COMPATIBILITY support for old logs
    if any(elem.name in ('vehicle_air_data', 'vehicle_magnetometer') for elem in data):
        baro_alt_meter_topic = 'vehicle_air_data'
        magnetometer_ga_topic = 'vehicle_magnetometer'
    else: # old
        baro_alt_meter_topic = 'sensor_combined'
        magnetometer_ga_topic = 'sensor_combined'
        
    if any(elem.name in 'estimator_wind' for elem in data):
        wind_estimator_topic = 'wind' #v1.14
    else:
        wind_estimator_topic = 'estimator_wind' #1.13
        
    manual_control_sp_controls = ['roll', 'pitch', 'yaw', 'throttle']
    manual_control_sp_throttle_range = '[-1, 1]'
    for topic in data:
        if topic.name == 'system_power':
            # COMPATIBILITY: rename fields to new format
            if 'voltage5V_v' in topic.data:     # old (prior to PX4/Firmware:213aa93)
                topic.data['voltage5v_v'] = topic.data.pop('voltage5V_v')
            if 'voltage3V3_v' in topic.data:    # old (prior to PX4/Firmware:213aa93)
                topic.data['sensors3v3[0]'] = topic.data.pop('voltage3V3_v')
            if 'voltage3v3_v' in topic.data:
                topic.data['sensors3v3[0]'] = topic.data.pop('voltage3v3_v')
        if topic.name == 'tecs_status':
            if 'airspeed_sp' in topic.data: # old (prior to PX4-Autopilot/pull/16585)
                topic.data['true_airspeed_sp'] = topic.data.pop('airspeed_sp')
        if topic.name == 'manual_control_setpoint':
            if 'throttle' not in topic.data: # old (prior to PX4-Autopilot/pull/15949)
                manual_control_sp_controls = ['y', 'x', 'r', 'z']
                manual_control_sp_throttle_range = '[0, 1]'

    if any(elem.name == 'vehicle_angular_velocity' for elem in data):
        rate_estimated_topic_name = 'vehicle_angular_velocity'
        rate_groundtruth_topic_name = 'vehicle_angular_velocity_groundtruth'
        rate_field_names = ['xyz[0]', 'xyz[1]', 'xyz[2]']
    else: # old
        rate_estimated_topic_name = 'vehicle_attitude'
        rate_groundtruth_topic_name = 'vehicle_attitude_groundtruth'
        rate_field_names = ['rollspeed', 'pitchspeed', 'yawspeed']
    if any(elem.name == 'manual_control_switches' for elem in data):
        manual_control_switches_topic = 'manual_control_switches'
    else: # old
        manual_control_switches_topic = 'manual_control_setpoint'
    dynamic_control_alloc = any(elem.name in ('actuator_motors', 'actuator_servos')
                                for elem in data)
    actuator_controls_0 = ActuatorControls(ulog, dynamic_control_alloc, 0)
    actuator_controls_1 = ActuatorControls(ulog, dynamic_control_alloc, 1)
    CONSTANTS_ONE_G = 9.80665
    M_PI_F          = 3.14159265

    # initialize flight mode changes
    flight_mode_changes = get_flight_mode_changes(ulog)

    # VTOL state changes & vehicle type
    vtol_states = None
    is_vtol = False
    is_vtol_tailsitter = False
    try:
        cur_dataset = ulog.get_dataset('vehicle_status')
        if np.amax(cur_dataset.data['is_vtol']) == 1:
            is_vtol = True
            # check if is tailsitter
            is_vtol_tailsitter = np.amax(cur_dataset.data['is_vtol_tailsitter']) == 1
            # find mode after transitions (states: 1=transition, 2=FW, 3=MC)
            if 'vehicle_type' in cur_dataset.data:
                vehicle_type_field = 'vehicle_type'
                vtol_state_mapping = {2: 2, 1: 3}
                vehicle_type = cur_dataset.data['vehicle_type']
                in_transition_mode = cur_dataset.data['in_transition_mode']
                vtol_states = []
                for i in range(len(vehicle_type)):
                    # a VTOL can change state also w/o in_transition_mode set
                    # (e.g. in Manual mode)
                    if i == 0 or in_transition_mode[i-1] != in_transition_mode[i] or \
                        vehicle_type[i-1] != vehicle_type[i]:
                        vtol_states.append((cur_dataset.data['timestamp'][i],
                                            in_transition_mode[i]))

            else: # COMPATIBILITY: old logs (https://github.com/PX4/Firmware/pull/11918)
                vtol_states = cur_dataset.list_value_changes('in_transition_mode')
                vehicle_type_field = 'is_rotary_wing'
                vtol_state_mapping = {0: 2, 1: 3}
            for i in range(len(vtol_states)):
                if vtol_states[i][1] == 0:
                    t = vtol_states[i][0]
                    idx = np.argmax(cur_dataset.data['timestamp'] >= t) + 1
                    vtol_states[i] = (t, vtol_state_mapping[
                        cur_dataset.data[vehicle_type_field][idx]])
            vtol_states.append((ulog.last_timestamp, -1))
    except (KeyError, IndexError) as error:
        vtol_states = None



    # Heading
    curdoc().template_variables['title_html'] = get_heading_html(
        ulog, px4_ulog, db_data, link_to_3d_page,
        additional_links=[("Open PID Analysis", link_to_pid_analysis_page)])

    # info text on top (logging duration, max speed, ...)
    curdoc().template_variables['info_table_html'] = \
        get_info_table_html(ulog, px4_ulog, db_data, vehicle_data, vtol_states)

    curdoc().template_variables['error_labels_html'] = get_error_labels_html()

    hardfault_html = get_hardfault_html(ulog)
    if hardfault_html is not None:
        curdoc().template_variables['hardfault_html'] = hardfault_html

    corrupt_log_html = get_corrupt_log_html(ulog)
    if corrupt_log_html:
        curdoc().template_variables['corrupt_log_html'] = corrupt_log_html

    # Position plot
    data_plot = DataPlot2D(data, plot_config, 'vehicle_local_position',
                           x_axis_label='[m]', y_axis_label='[m]', plot_height='large')
    data_plot.add_graph('y', 'x', colors2[0], 'Estimated',
                        check_if_all_zero=True)
    if not data_plot.had_error: # vehicle_local_position is required
        data_plot.change_dataset('vehicle_local_position_setpoint')
        data_plot.add_graph('y', 'x', colors2[1], 'Setpoint')
        # groundtruth (SITL only)
        data_plot.change_dataset('vehicle_local_position_groundtruth')
        data_plot.add_graph('y', 'x', color_gray, 'Groundtruth')
        # GPS + position setpoints
        plot_map(ulog, plot_config, map_type='plain', setpoints=True,
                 bokeh_plot=data_plot.bokeh_plot)
        if data_plot.finalize() is not None:
            plots.append(data_plot.bokeh_plot)

    if any(elem.name == 'vehicle_gps_position' for elem in ulog.data_list):
        # Leaflet Map
        try:
            pos_datas, flight_modes = ulog_to_polyline(ulog, flight_mode_changes)
            curdoc().template_variables['pos_datas'] = pos_datas
            curdoc().template_variables['pos_flight_modes'] = flight_modes
        except:
            pass
        curdoc().template_variables['has_position_data'] = True

    # initialize parameter changes
    changed_params = None
    if not 'replay' in ulog.msg_info_dict: # replay can have many param changes
        if len(ulog.changed_parameters) > 0:
            changed_params = ulog.changed_parameters
            plots.append(None) # save space for the param change button

    ### Add all data plots ###

    x_range_offset = (ulog.last_timestamp - ulog.start_timestamp) * 0.05
    x_range = Range1d(ulog.start_timestamp - x_range_offset, ulog.last_timestamp + x_range_offset)

    # Altitude estimate
    try:
        baro_data = ulog.get_dataset('vehicle_air_data')
        baro_time = baro_data.data['timestamp']
        ekf_data = ulog.get_dataset('estimator_states')
        innov_data = ulog.get_dataset('estimator_innovations')
        
        baro_alt        = baro_data.data['baro_alt_meter']
        ekf_alt         = np.interp(baro_time, ekf_data.data['timestamp'], ekf_data.data['states[9]'])
        baro_innovation = np.interp(baro_time, innov_data.data['timestamp'], innov_data.data['baro_vpos'])
        baro_hgt_offset = ekf_alt + baro_alt - baro_innovation
        offset_baro_alt = baro_alt - baro_hgt_offset
        
        data_plot = DataPlot(data, plot_config, 'vehicle_gps_position',
                             y_axis_label='[m]', title='Altitude Estimate',
                             changed_params=changed_params, x_range=x_range)
        data_plot.add_graph([lambda data: ('alt', data['alt']*0.001)],
                            colors8[0:1], ['GPS Altitude'])
        data_plot.change_dataset(baro_alt_meter_topic)
        data_plot.add_graph(['baro_alt_meter'], colors8[1:2], ['Barometer Altitude'])
        data_plot.change_dataset('vehicle_global_position')
        data_plot.add_graph(['alt'], colors8[2:3], ['Vehicle Global Position'])
        data_plot.change_dataset('position_setpoint_triplet')
        data_plot.add_circle(['current.alt'], [plot_config['mission_setpoint_color']],
                             ['Altitude Setpoint'])
        data_plot.change_dataset(actuator_controls_0.thrust_sp_topic)
        data_plot.add_graph([lambda data: ('thrust', actuator_controls_0.thrust_x * 100.0)],
                            colors8[6:7], ['Thrust [0, 100]'])
        data_plot.change_dataset('vehicle_air_data')
        data_plot.add_graph([lambda data: ('baro_alt_meter', baro_hgt_offset)],
                            colors8[7:8], ['Baro Height Offset'])
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
        
        if data_plot.finalize() is not None: plots.append(data_plot)

    except:
        pass

    # Altitude change estimate
    # Estimate the gap between the barometer altitude and the local altitude estimate
    baro_local_alt_gap = 0
    baro_local_alt_rms = 0
    try:
        pos_set = ulog.get_dataset('vehicle_local_position')
        ref_time = pos_set.data['timestamp']
        ref_data = -(pos_set.data['z'] - pos_set.data['z'][0])
        baro_alt_set = ulog.get_dataset(baro_alt_meter_topic)
        baro_time = baro_alt_set.data['timestamp']
        baro_alt  = baro_alt_set.data['baro_alt_meter'] - baro_alt_set.data['baro_alt_meter'][0]
        
        (baro_local_alt_gap, baro_local_alt_rms, baro_diff) = get_mean_gap(ref_time, ref_data, baro_time, baro_alt)
    
    except (KeyError, IndexError) as error:
        print('Could not compute baro_local_alt_gap')
        
    data_plot = DataPlot(data, plot_config, baro_alt_meter_topic,
                         y_axis_label='[m]', title='Altitude Change Estimate',
                         changed_params=changed_params, x_range=x_range)
    data_plot.change_dataset('vehicle_gps_position')
    data_plot.add_graph([lambda data: ('alt', (data['alt'] - data['alt'][0])*0.001)],
                        colors8[0:1], ['GPS Altitude'])
    data_plot.change_dataset(baro_alt_meter_topic)
    data_plot.add_graph([lambda data: ('baro_alt_meter', data['baro_alt_meter'] - data['baro_alt_meter'][0] - baro_local_alt_gap)],
                         colors8[1:2], [('Baro; gap: mean=%.2f;RMS=%.2f') % (baro_local_alt_gap, baro_local_alt_rms)])
    data_plot.change_dataset('distance_sensor')
    data_plot.change_dataset('vehicle_local_position')
    data_plot.add_graph([lambda data: ('z', -(data['z'] - data['z'][0]))],
                         colors8[3:4], ['Vehicle Local Position'])
    data_plot.add_graph([lambda data: ('vz', integrate.cumtrapz(-data['vz'], data['timestamp'] * 1e-6))],
                                      colors8[5:6], ['Integrated climb rate'])
    data_plot.change_dataset('distance_sensor')
    data_plot.add_graph(['current_distance'], colors8[6:7],
                        ['Distance Sensor'])
    
    data_plot.change_dataset('vehicle_local_position')
    data_plot.add_graph(['dist_bottom'], colors8[4:5], ['Distance to bottom'], use_step_lines=True)
    
    try:
        starting_local_altitude = ulog.get_dataset('vehicle_local_position').data['z'][0]
        data_plot.change_dataset('vehicle_local_position_setpoint')
        data_plot.add_graph([lambda data: ('z', -(data['z'] - starting_local_altitude))], colors8[7:8],
                            ['Z Setpoint'])
        
    except (KeyError, IndexError) as error:
        print('Error trying to get starting local altitude: '+str(error))
    
    try:
        data_plot.change_dataset('vehicle_air_data')
        data_plot.add_graph([lambda data: ('baro_alt_meter', offset_baro_alt + ekf_data.data['states[9]'][0])], colors13[8:9],
                            ['Baro with offset'])
    
        
    except:
        pass
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

    if data_plot.finalize() is not None: plots.append(data_plot)


    # VTOL tailistter orientation conversion, if relevant
    if is_vtol_tailsitter:
        [tailsitter_attitude, tailsitter_rates] = tailsitter_orientation(ulog, vtol_states)

    # Roll/Pitch/Yaw angle & angular rate
    for index, axis in enumerate(['roll', 'pitch', 'yaw']):
        # angle
        axis_name = axis.capitalize()
        data_plot = DataPlot(data, plot_config, 'vehicle_attitude',
                             y_axis_label='[deg]', title=axis_name+' Angle',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        if is_vtol_tailsitter:
            if tailsitter_attitude[axis] is not None:
                data_plot.add_graph([lambda data: (axis+'_q',
                                                   np.rad2deg(tailsitter_attitude[axis]))],
                                    colors3[0:1], [axis_name+' Estimated'], mark_nan=True)
        else:
            data_plot.add_graph([lambda data: (axis, np.rad2deg(data[axis]))],
                                colors3[0:1], [axis_name+' Estimated'], mark_nan=True)

        data_plot.change_dataset('vehicle_attitude_setpoint')
        data_plot.add_graph([lambda data: (axis+'_d', np.rad2deg(data[axis+'_d']))],
                            colors3[1:2], [axis_name+' Setpoint'],
                            use_step_lines=True)
        if axis == 'yaw':
            data_plot.add_graph(
                [lambda data: ('yaw_sp_move_rate', np.rad2deg(data['yaw_sp_move_rate']))],
                colors3[2:3], [axis_name+' FF Setpoint [deg/s]'],
                use_step_lines=True)
            
            data_plot.change_dataset('position_controller_status')
            data_plot.add_graph([lambda data: ('nav_bearing',
                                               np.rad2deg(data['nav_bearing'])),
                                 lambda data: ('target_bearing',
                                               np.rad2deg(data['target_bearing']))],
                                colors8[3:5], ['L1 Nav Bearing', 'L1 Target Bearing'], mark_nan=True)
            data_plot.change_dataset('vehicle_attitude_setpoint')
            data_plot.add_graph(['fw_control_yaw'], colors8[5:6],
                                ['Control yaw active'])
        elif axis == 'roll':
            data_plot.change_dataset('vehicle_attitude_setpoint')
            data_plot.add_graph(
                [lambda data: ('stall_prevention_roll_limit', np.rad2deg(data['stall_prevention_roll_limit']))],
                colors3[2:3], ['Stall Prevention ' + axis_name+' limit'],
                use_step_lines=True)
            data_plot.change_dataset('npfg_status')
            data_plot.add_graph(
                [lambda data: ('roll_limit_deg', data['roll_limit_deg'])],
                colors3[2:3], ['Stall Prevention ' + axis_name+' limit'],
                use_step_lines=True)
            data_plot.add_graph(
                [lambda data: ('roll_limit_deg', -data['roll_limit_deg'])],
                colors3[2:3], ['Stall Prevention ' + axis_name+' limit'],
                use_step_lines=True)
            
            
            
        data_plot.change_dataset('vehicle_attitude_groundtruth')
        data_plot.add_graph([lambda data: (axis, np.rad2deg(data[axis]))],
                            [color_gray], [axis_name+' Groundtruth'])
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

        if data_plot.finalize() is not None: plots.append(data_plot)

        # rate
        data_plot = DataPlot(data, plot_config, rate_estimated_topic_name,
                             y_axis_label='[deg/s]', title=axis_name+' Angular Rate',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        if is_vtol_tailsitter:
            if tailsitter_rates[axis] is not None:
                data_plot.add_graph([lambda data: (axis+'_q',
                                                   np.rad2deg(tailsitter_rates[axis]))],
                                    colors3[0:1], [axis_name+' Rate Estimated'], mark_nan=True)
        else:
            data_plot.add_graph([lambda data: (axis+'speed',
                                               np.rad2deg(data[rate_field_names[index]]))],
                                colors3[0:1], [axis_name+' Rate Estimated'], mark_nan=True)
        data_plot.change_dataset('vehicle_rates_setpoint')
        data_plot.add_graph([lambda data: (axis, np.rad2deg(data[axis]))],
                            colors3[1:2], [axis_name+' Rate Setpoint'],
                            mark_nan=True, use_step_lines=True)
        axis_letter = axis[0].upper()
        rate_int_limit = '(*100)'
        # this param is MC/VTOL only (it will not exist on FW)
        rate_int_limit_param = 'MC_' + axis_letter + 'R_INT_LIM'
        if rate_int_limit_param in ulog.initial_parameters:
            rate_int_limit = '[-{0:.0f}, {0:.0f}]'.format(
                ulog.initial_parameters[rate_int_limit_param]*100)
        data_plot.change_dataset('rate_ctrl_status')
        data_plot.add_graph([lambda data: (axis, data[axis+'speed_integ']*100)],
                            colors3[2:3], [axis_name+' Rate Integral '+rate_int_limit])
        data_plot.change_dataset(rate_groundtruth_topic_name)
        data_plot.add_graph([lambda data: (axis+'speed',
                                           np.rad2deg(data[rate_field_names[index]]))],
                            [color_gray], [axis_name+' Rate Groundtruth'])
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

        if data_plot.finalize() is not None: plots.append(data_plot)

    # Local position
    try:
        gps_t = ulog.get_dataset('vehicle_gps_position').data['timestamp']
        gps_x, gps_y = map_projection_from_ulog(ulog)
    except (KeyError, IndexError) as error:
        print('map_projection_from_ulog error : '+str(error))
        
    for axis in ['x', 'y', 'z']:
        data_plot = DataPlot(data, plot_config, 'vehicle_local_position',
                             y_axis_label='[m]', title='Local Position '+axis.upper(),
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        data_plot.add_graph([axis], colors2[0:1], [axis.upper()+' Estimated'], mark_nan=True)
        data_plot.change_dataset('vehicle_local_position_setpoint')
        data_plot.add_graph([axis], colors2[1:2], [axis.upper()+' Setpoint'],
                            use_step_lines=True)
        if axis == 'x':
            data_plot.change_dataset('vehicle_gps_position')
            try:
                data_plot.add_graph([lambda data: ('lat',
                                                   gps_x)],
                                    colors3[2:3], ['GPS_' + axis])
            except:
                pass
            
        if axis == 'y':
            data_plot.change_dataset('vehicle_gps_position')
            try:
                data_plot.add_graph([lambda data: ('lon',
                                                   gps_y)],
                                    colors3[2:3], ['GPS_' + axis])
            except:
                pass
            
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

        if data_plot.finalize() is not None: plots.append(data_plot)



    # Velocity
    data_plot = DataPlot(data, plot_config, 'vehicle_local_position',
                         y_axis_label='[m/s]', title='Velocity',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['vx', 'vy', 'vz'], colors8[0:3], ['X', 'Y', 'Z'])
    data_plot.add_graph([lambda data: ('z', (data['z'][1:] - data['z'][:-1])/(data['timestamp'][1:] - data['timestamp'][:-1]) * 1e6)],
                                      colors8[7:8], ['dz/dt'])
    data_plot.change_dataset('vehicle_local_position_setpoint')
    data_plot.add_graph(['vx', 'vy', 'vz'], [colors8[5], colors8[4], colors8[6]],
                        ['X Setpoint', 'Y Setpoint', 'Z Setpoint'], use_step_lines=True)

    data_plot.change_dataset('vehicle_gps_position')
    data_plot.add_graph(['vel_n_m_s', 'vel_e_m_s', 'vel_d_m_s'], colors13[8:11], ['GPS N', 'GPS E', 'GPS D'])
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

    if data_plot.finalize() is not None: plots.append(data_plot)


    # Visual Odometry (only if topic found)
    if any(elem.name == 'vehicle_visual_odometry' for elem in data):
        # Vision position
        data_plot = DataPlot(data, plot_config, 'vehicle_visual_odometry',
                             y_axis_label='[m]', title='Visual Odometry Position',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        data_plot.add_graph(['x', 'y', 'z'], colors3, ['X', 'Y', 'Z'], mark_nan=True)
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

        data_plot.change_dataset('vehicle_local_position_groundtruth')
        data_plot.add_graph(['x', 'y', 'z'], colors8[2:5],
                            ['Groundtruth X', 'Groundtruth Y', 'Groundtruth Z'])

        if data_plot.finalize() is not None: plots.append(data_plot)


        # Vision velocity
        data_plot = DataPlot(data, plot_config, 'vehicle_visual_odometry',
                             y_axis_label='[m]', title='Visual Odometry Velocity',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        data_plot.add_graph(['vx', 'vy', 'vz'], colors3, ['X', 'Y', 'Z'], mark_nan=True)
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

        data_plot.change_dataset('vehicle_local_position_groundtruth')
        data_plot.add_graph(['vx', 'vy', 'vz'], colors8[2:5],
                            ['Groundtruth VX', 'Groundtruth VY', 'Groundtruth VZ'])
        if data_plot.finalize() is not None: plots.append(data_plot)


        # Vision attitude
        data_plot = DataPlot(data, plot_config, 'vehicle_visual_odometry',
                             y_axis_label='[deg]', title='Visual Odometry Attitude',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        data_plot.add_graph([lambda data: ('roll', np.rad2deg(data['roll'])),
                             lambda data: ('pitch', np.rad2deg(data['pitch'])),
                             lambda data: ('yaw', np.rad2deg(data['yaw']))],
                            colors3, ['Roll', 'Pitch', 'Yaw'], mark_nan=True)
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

        data_plot.change_dataset('vehicle_attitude_groundtruth')
        data_plot.add_graph([lambda data: ('roll', np.rad2deg(data['roll'])),
                             lambda data: ('pitch', np.rad2deg(data['pitch'])),
                             lambda data: ('yaw', np.rad2deg(data['yaw']))],
                            colors8[2:5],
                            ['Roll Groundtruth', 'Pitch Groundtruth', 'Yaw Groundtruth'])

        if data_plot.finalize() is not None: plots.append(data_plot)

        # Vision attitude rate
        data_plot = DataPlot(data, plot_config, 'vehicle_visual_odometry',
                             y_axis_label='[deg]', title='Visual Odometry Attitude Rate',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        data_plot.add_graph([lambda data: ('rollspeed', np.rad2deg(data['rollspeed'])),
                             lambda data: ('pitchspeed', np.rad2deg(data['pitchspeed'])),
                             lambda data: ('yawspeed', np.rad2deg(data['yawspeed']))],
                            colors3, ['Roll Rate', 'Pitch Rate', 'Yaw Rate'], mark_nan=True)
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

        data_plot.change_dataset(rate_groundtruth_topic_name)
        data_plot.add_graph([lambda data: ('rollspeed', np.rad2deg(data[rate_field_names[0]])),
                             lambda data: ('pitchspeed', np.rad2deg(data[rate_field_names[1]])),
                             lambda data: ('yawspeed', np.rad2deg(data[rate_field_names[2]]))],
                            colors8[2:5],
                            ['Roll Rate Groundtruth', 'Pitch Rate Groundtruth',
                             'Yaw Rate Groundtruth'])

        if data_plot.finalize() is not None: plots.append(data_plot)

        # Vision latency
        data_plot = DataPlot(data, plot_config, 'vehicle_visual_odometry',
                             y_axis_label='[ms]', title='Visual Odometry Latency',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        data_plot.add_graph(
            [lambda data: ('latency', 1e-3*(data['timestamp'] - data['timestamp_sample']))],
            colors3, ['VIO Latency'], mark_nan=True)
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

        if data_plot.finalize() is not None: plots.append(data_plot)


    # Mission result
    data_plot = DataPlot(data, plot_config, 'mission_result',
                         y_axis_label='', title='Mission result',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['instance_count', 'seq_reached', 'seq_current',
                         'seq_total', 'item_changed_index',
                         'finished', 'item_do_jump_changed', 'execution_mode'], 
                        colors8[0:8], ['Mission check count', 'Sequence Reached', 'Sequence Current',
                                       'Sequence Total', 'Item Changed Index', 'Finished',
                                       'Item do_jump changed', 'Execution Mode'], use_step_lines=True)
    data_plot.change_dataset('vehicle_command')
    data_plot.add_circle([lambda data: ('command', data['command'] == 176)], [plot_config['mission_setpoint_color']],
                         ['DO_SET_MODE'])
    data_plot.add_circle([lambda data: ('command', data['command'] == 195)], [colors8[0]],
                         ['DO_SET_ROI_LOCATION'])
    data_plot.change_dataset('position_setpoint_triplet')
    data_plot.add_circle(['current.type'], [colors8[1]], ['Setpoint Type'])
    # data_plot.change_dataset('vehicle_command')
    # data_plot.add_graph(['command', 'param1', 'param2', 'param3', 'source_system'],
    #                     colors8[0:5], ['Command', 'Param 1', 'Param 2', 'Param 3', 'Source System'], use_step_lines=True)
    # data_plot.change_dataset('vehicle_local_position_setpoint')
    # data_plot.add_graph(['vx', 'vy', 'vz'], [colors8[5], colors8[4], colors8[6]],
    #                     ['X Setpoint', 'Y Setpoint', 'Z Setpoint'], use_step_lines=True)
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    # Land detector status
    data_plot = DataPlot(data, plot_config, 'vehicle_land_detected',
                         y_axis_label='', title='Landing detector status',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['freefall', 'ground_contact', 'maybe_landed',
                         'landed', 'in_ground_effect',
                         'in_descend', 'has_low_throttle', 'vertical_movement',
                         'horizontal_movement', 'close_to_ground_or_skipped_check',
                         'at_rest'], 
                        colors13[0:11], 
                        ['Freefall', 'Ground Contact', 'Maybe Landed',
                         'Landed', 'In Ground Effect',
                         'In Descent', 'Has Low Throttle', 'Vertical Movement',
                         'Horizontal Movement', 'Close to ground or skipped check',
                         'At Rest'], use_step_lines=True)
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    
    # Landing sequence status
    data_plot = DataPlot(data, plot_config, 'actuator_controls_0',
                         y_start=0, title='Landing Sequence',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['control[4]', 'control[5]', 'control[6]'],
                        colors8[0:3], ['Flaps', 'Spoilers', 'Brakes'], use_step_lines=True)
    try:
        land_hgt_trigger       = ulog.initial_parameters['FW_LND_GEAR_HGT']
        data_plot.change_dataset('vehicle_local_position')
        data_plot.add_graph([lambda data: ('dist_bottom',
                                       data['dist_bottom'] < land_hgt_trigger)], 
                            colors8[3:4], [('Distance to bottom below %.2f [m] (landing gear height)') % (land_hgt_trigger)], 
                            use_step_lines=True)
    except (KeyError, IndexError) as error:
        print('FW_LND_GEAR_HGT does not exist: '+str(error))
        
    try:
        stall_airspeed_mps         = ulog.initial_parameters['FW_AIRSPD_STALL']
        spoiler_speed_scalar       = ulog.initial_parameters['FW_LND_SPSPD_SC']
        brake_speed_scalar         = ulog.initial_parameters['FW_LND_BKSPD_SC']
        
        data_plot.change_dataset('vehicle_local_position')
        data_plot.add_graph([lambda data: ('groundspeed_estimated',
                                           np.sqrt(data['vx']**2 + data['vy']**2) < stall_airspeed_mps * spoiler_speed_scalar)],
                            colors8[4:5], [('Ground Speed below %.2f [m/s] (spoilers)') % (stall_airspeed_mps * spoiler_speed_scalar)],
                            use_step_lines=True)
        data_plot.add_graph([lambda data: ('groundspeed_estimated',
                                           np.sqrt(data['vx']**2 + data['vy']**2) < stall_airspeed_mps * brake_speed_scalar)],
                            colors8[5:6], [('Ground Speed below %.2f [m/s] (brakes)') % (stall_airspeed_mps * brake_speed_scalar)],
                            use_step_lines=True)
    except (KeyError, IndexError) as error:
        print('FW_LND_SPSPD_SC or FW_AIRSPD_STALL does not exist: '+str(error))
    
    data_plot.change_dataset('vehicle_attitude_setpoint')
    data_plot.add_graph(['fw_control_yaw'], colors8[7:8],
                        ['Control yaw active'], use_step_lines=True)
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    
    # Landing status
    data_plot = DataPlot(data, plot_config, 'position_controller_landing_status',
                         y_start=0, title='Landing Status',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['land_onslope', 'land_noreturn_vertical', 'land_noreturn_horizontal',
                         'land_motor_lim', 'land_spoilers', 'land_brake'],
                        colors8[0:6], ['on slope', 'no return vertical', 'no return horizontal',
                                             'motor lim', 'spoilers', 'brake'], mark_nan=True, use_step_lines=True)
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    '''
    #try:
    gps_t = ulog.get_dataset('vehicle_gps_position').data['timestamp']
    gps_x, gps_y = map_projection_from_ulog(ulog)
    gps_ds = np.sqrt((np.diff(gps_x))**2 + (np.diff(gps_y))**2)
    ground_course = np.cumsum(gps_ds)
    
    home_set    = ulog.get_dataset('home_position')
    lat_rad     = np.deg2rad(home_set.data['lat'])
    lon_rad     = np.deg2rad(home_set.data['lon'])
    home_x, home_y = project_latlon_on_map(ulog, lat_rad, lon_rad)
    home_disp   = np.sqrt((home_x - gps_x[0])**2 + (home_y - gps_y[0])**2)
    
    wp_set      = ulog.get_dataset('position_setpoint_triplet')
    lat_rad     = np.deg2rad(wp_set.data['current.lat'])
    lon_rad     = np.deg2rad(wp_set.data['current.lon'])
    wp_x, wp_y  = project_latlon_on_map(ulog, lat_rad, lon_rad)
    wp_disp     = np.sqrt((wp_x - gps_x[0])**2 + (wp_y - gps_y[0])**2)
    
    wp_interp_x = interpolate.interp1d(wp_set.data['timestamp'], wp_x, kind = 'previous', fill_value = "extrapolate")
    wp_interp_y = interpolate.interp1d(wp_set.data['timestamp'], wp_y, kind = 'previous', fill_value = "extrapolate")
    wp_dist     = np.sqrt((wp_interp_x(gps_t) - gps_x)**2 + (wp_interp_y(gps_t) - gps_y)**2)
    #except:
     #   print('No GPS data')
      #  pass
    #try:
      '''  
    try:
        reg_plot = altitude_profile_plot(ulog, plot_config, 'Altitude Profile', 'Distance Traveled [m]', 'Altitude [m]')
    
        plots.append(reg_plot)
        
    except:
        pass
    

    # Glide angle
    data_plot = DataPlot(data, plot_config, 'vehicle_local_position',
                         y_start=0, title='Glide angles',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph([lambda data: ('groundspeed_estimated',
                                        180.0/np.pi * np.arctan2(-data['vz'] * (np.sqrt(data['vx']**2 + data['vy']**2) > 1.0 ), 
                                        np.sqrt(data['vx']**2 + data['vy']**2) ) )],
                            colors8[0:1], ['Glide angle'],
                            use_step_lines=True)
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    
    # L1 - Waypoint distance
    data_plot = DataPlot(data, plot_config, 'position_controller_status', title='L1 Status',
                         y_axis_label='[m]', plot_height='small',
                         changed_params=changed_params, x_range=x_range)
    data_plot.add_graph(['wp_dist', 
                         'xtrack_error'],
                        colors8[0:2], 
                        ['Waypoint Distance',
                         'Crosstrack Error'],
                        mark_nan=True)
    try:
        L1_damping       = ulog.initial_parameters['FW_L1_DAMPING']
        L1_period        = ulog.initial_parameters['FW_L1_PERIOD']
        L1_ratio         = 1.0 / M_PI_F * L1_damping * L1_period
        # Plot L1 Distance
        data_plot.change_dataset('vehicle_gps_position')
        data_plot.add_graph([lambda data: ('vel_m_s',
                                           data['vel_m_s'] * L1_ratio)],
                            colors8[2:3], [('L1 Distance (L1 ratio = %.2f)') % (L1_ratio)])
    except (KeyError, IndexError) as error:
        pass
    data_plot.change_dataset('position_controller_status')
    data_plot.add_graph(['acceptance_radius'],
                        colors8[3:4], 
                        ['Acceptance Radius'],
                        mark_nan=True)
    
    data_plot.change_dataset('vehicle_imu_status')
    data_plot.add_graph(['accel_vibration_metric'], colors8[4:5],
                         ['Accel 0 Vibration Level [m/s^2]'])
    
    data_plot.change_dataset('vehicle_attitude_setpoint')
    data_plot.add_graph(['fw_control_yaw'], colors8[5:6],
                        ['Control yaw active'])
        
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)


    # Wind speed
    data_plot = DataPlot(data, plot_config, wind_estimator_topic, title='Wind Estimates',
                         y_axis_label='[m/s]', plot_height='small',
                         changed_params=changed_params, x_range=x_range,
                         topic_instance=0)
    data_plot.add_graph(['windspeed_north', 
                         'windspeed_east',
                         lambda data: ('beta_innov', np.rad2deg(data['beta_innov'])),
                         lambda data: ('variance_east', np.sqrt(
                             data['windspeed_north']**2 + data['windspeed_east']**2)),
                         lambda data: ('tas_innov', 
                                       np.rad2deg(np.arctan2(data['windspeed_east'], data['windspeed_north'])))],
                        colors8[0:5], 
                        ['Wind Due North', 
                         'Wind Due East',
                         'Sideslip angle',
                         'Wind Strength', 
                         'Wind Direction due North'],
                        mark_nan=True)
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    # Angle of attack and sideslip angle
    try:
        data_plot = DataPlot(data, plot_config, 'vehicle_attitude', title='Aero Angles',
                              y_axis_label='[deg]', plot_height='small',
                              changed_params=changed_params, x_range=x_range,
                              topic_instance=0)
        AoA, SSA, AoA_noWind, SSA_noWind = compute_aero_angles_log(ulog)
        
        data_plot.add_graph([lambda data: ('q[0]', np.rad2deg(AoA)),
                              lambda data: ('q[1]', np.rad2deg(AoA_noWind)),
                            lambda data: ('q[2]', np.rad2deg(SSA)),
                            lambda data: ('q[3]', np.rad2deg(SSA_noWind))],
                              colors8[0:4], 
                            ['AOA', 'AOA no wind',
                              'SSA', 'SSA no wind'],
                            mark_nan=True)
        
        data_plot.change_dataset(wind_estimator_topic)
        data_plot.add_graph([lambda data: ('beta_innov', np.rad2deg(data['beta_innov']))],
                            colors8[4:5], 
                            ['SSA Est'],
                            mark_nan=True)
        
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
        if data_plot.finalize() is not None: plots.append(data_plot)
    except:
        print('Could not compute AoA and SSA')
        pass
    
    
    try:
        gs = ulog.get_dataset('vehicle_local_position')
        ground_speed = np.sqrt(gs.data['vx']**2 + gs.data['vy']**2)
        ground_accel = np.concatenate((np.zeros(1), np.diff(ground_speed)/np.diff(gs.data['timestamp']) * 1e6))
        
    except:
        print('Could not compute ground_speed and ground_accel')
        pass
    
    # Airspeed vs Ground speed: but only if there's valid airspeed data or a VTOL
    try:
        if is_vtol or ulog.get_dataset('airspeed') is not None:
            data_plot = DataPlot(data, plot_config, 'vehicle_local_position',
                                 y_axis_label='[m/s]', title='Airspeed',
                                 plot_height='small',
                                 changed_params=changed_params, x_range=x_range)
            data_plot.add_graph([lambda data: ('groundspeed_estimated',
                                               ground_speed)],
                                colors8[0:1], ['Ground Speed Estimated'])
            data_plot.add_graph([lambda data: ('groundspeed_estimated',
                                               ground_accel)],
                                colors13[8:9], ['Ground Accel Estimated [m/s/s]'])
            
            if any(elem.name == 'airspeed_validated' for elem in data):
                airspeed_validated = ulog.get_dataset('airspeed_validated')
                data_plot.change_dataset('airspeed_validated')
                if np.amax(airspeed_validated.data['airspeed_sensor_measurement_valid']) == 1:
                    data_plot.add_graph(['true_airspeed_m_s'], colors8[1:2],
                                        ['True Airspeed'])
                else:
                    data_plot.add_graph(['true_ground_minus_wind_m_s'], colors8[1:2],
                                        ['True Airspeed (estimated)'])

            data_plot.change_dataset('airspeed')
            data_plot.add_graph(['indicated_airspeed_m_s'], colors13[9:10],
                                ['Indicated Airspeed'])
            data_plot.change_dataset('airspeed_validated')
            data_plot.add_graph(['calibrated_airspeed_m_s'], colors13[10:11],
                                ['Calibrated Airspeed'])
            data_plot.change_dataset('vehicle_gps_position')
            data_plot.add_graph(['vel_m_s'], colors8[2:3], ['Ground Speed (from GPS)'])
            data_plot.change_dataset('tecs_status')
            data_plot.add_graph(['true_airspeed_sp'], colors8[3:4], ['True Airspeed Setpoint'])
            data_plot.add_graph(['true_airspeed_filtered'], colors8[4:5], ['True Airspeed Filtered'])
            data_plot.add_graph(['true_airspeed_innovation'], colors8[5:6], ['True Airspeed Innovation'])
            
            max_speed       = ulog.initial_parameters['FW_AIRSPD_MAX']
            try:
                max_throttle    = ulog.initial_parameters['FW_THR_MAX']
            except:
                max_throttle    = 1.0
                
            if max_throttle < 0.1:
                max_throttle = 1.0
                
            throttle_scale  = max_speed / np.max([max_throttle, 1e-6])
            
            data_plot.add_graph([lambda data: ('throttle_sp', data['throttle_sp'] * throttle_scale),
                                 lambda data: ('throttle_integ', data['throttle_integ'] * throttle_scale)],
                                colors8[6:8], [('TECS Throttle Setpoint * %.2f') % (throttle_scale), 
                                               ('TECS Throttle Integral * %.2f') % (throttle_scale)])
            data_plot.change_dataset(actuator_controls_0.torque_sp_topic)
            data_plot.add_graph([lambda data: ('thrust', actuator_controls_0.thrust_x * throttle_scale)],
                                colors8[3:4], [('Throttle * %.2f') % (throttle_scale)], mark_nan=True)
            
            plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    
            if data_plot.finalize() is not None: plots.append(data_plot)
    except (KeyError, IndexError) as error:
        pass

    
    # Airspeed sensor calibration quality
    # try:
    data_plot = DataPlot(data, plot_config, 'vehicle_local_position', title='Airspeed Diagnostics',
                          y_axis_label='[m/s]', plot_height='small',
                          changed_params=changed_params, x_range=x_range,
                          topic_instance=0)
    # AoA, SSA, AoA_noWind, SSA_noWind = compute_aero_angles_log(ulog)
    
    
    data_plot.add_graph([lambda data: ('groundspeed_estimated',
                                       ground_speed)],
                        colors13[0:1], ['Ground Speed'])
    
    data_plot.change_dataset('airspeed')
    ts = ulog.get_dataset('airspeed')
    airspeed = ts.data['indicated_airspeed_m_s']
    airspeed_scale = 1.0
    data_plot.add_graph([lambda data: ('true_airspeed_derivative',
                                       airspeed_scale * (airspeed - airspeed[0]))],
                        colors8[1:2], 
                        ['Scaled Airspeed; SCALE = %.2f ' % airspeed_scale],
                        mark_nan=True)
    
    data_plot.change_dataset('vehicle_attitude_setpoint')
    data_plot.add_graph(['fw_control_yaw_wheel'], colors8[2:3], ['Wheel Control'])
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    # except:
    #     print('Could not compute airspeed diagnostics')
    #     pass

    # Fuze control diagnostics
    data_plot = DataPlot(data, plot_config, 'fuze_control_status',
                         y_start=0, y_axis_label='', title='Fuze Control Status',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    
    data_plot.add_graph(['state_dist_from_home', 'arm_state', 'arm_control_output', 'arm_firing_measure',
                         'arm_override', 'safety_actuator_armed', 'safety_timer', 'safety_airspeed_high', 
                         'safety_home_distance', 'state_dist_from_home_valid'], colors13[0:10], 
                        ['dist from home', 'arm state', 'arm control output', 'arm control feedback',
                         'arm override', 'safety actuator armed', 'safety timer', 'safety airspeed', 
                         'safety home distance', 'state dist from home valid'])
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    # TECS - airspeed derivative
    data_plot = DataPlot(data, plot_config, 'tecs_status', title='TECS Airspeed Derivative',
                         y_axis_label='[m/s/s]', plot_height='small',
                         changed_params=changed_params, x_range=x_range)
    data_plot.add_graph(['true_airspeed_derivative_raw', 
                         'true_airspeed_derivative', 
                         'true_airspeed_derivative_sp'],
                        colors8[0:3], 
                        ['True Airspeed Derivative Raw', 
                         'True Airspeed Derivative Filtered',
                         'True Airspeed Derivative Setpoint'],
                        mark_nan=True)
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    
    # TECS (fixed-wing or VTOLs)
    data_plot = DataPlot(data, plot_config, 'tecs_status', title='TECS Altitude',
                         y_axis_label='[m/s]', plot_height='small',
                         changed_params=changed_params, x_range=x_range)
    data_plot.add_graph(['altitude_filtered', 'altitude_sp'],
                        colors2, ['Height', 'Height Setpoint'],
                        mark_nan=True)
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    # TECS climb rate (fixed-wing or VTOLs)
    data_plot = DataPlot(data, plot_config, 'tecs_status', y_start=0, title='TECS Climb Rate',
                         y_axis_label='[m/s]', plot_height='small',
                         changed_params=changed_params, x_range=x_range)
    data_plot.add_graph(['height_rate', 'height_rate_setpoint'],
                        colors2, ['Height Rate', 'Height Rate Setpoint'],
                        mark_nan=True)
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)

    # TECS - energy
    data_plot = DataPlot(data, plot_config, 'tecs_status', title='TECS Energy',
                         y_axis_label='[m/s/s]', plot_height='small',
                         changed_params=changed_params, x_range=x_range)
    data_plot.add_graph(['total_energy', 
                         'total_energy_sp',
                         'total_energy_balance',
                         'total_energy_balance_sp'],
                        colors8[0:4], 
                        ['Total Energy', 
                         'Total Energy Setpoint',
                         'Energy Balance', 
                         'Energy Balance Setpoint'],
                        mark_nan=True)
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    
    # TECS - energy rates
    try:
        min_sink_rate       = ulog.initial_parameters['FW_T_SINK_MIN']
        max_climb_rate      = ulog.initial_parameters['FW_T_CLMB_MAX']
        
        STE_rate_max =   max_climb_rate * CONSTANTS_ONE_G;
        STE_rate_min = - min_sink_rate  * CONSTANTS_ONE_G;
        
        throttle_to_STE_rate = (STE_rate_max - STE_rate_min)
        
        data_plot = DataPlot(data, plot_config, 'tecs_status', title='TECS Energy Rates',
                             y_axis_label='', plot_height='small',
                             changed_params=changed_params, x_range=x_range)
        data_plot.add_graph(['total_energy_rate',
                             'total_energy_rate_sp'],
                            colors8[0:2], 
                            ['Total Energy Rate', 
                             'Total Energy Rate Setpoint'],
                            mark_nan=True)
        
        data_plot.add_graph([lambda data: ('throttle_integ', data['throttle_integ'] * throttle_to_STE_rate)],
                            colors8[2:3], [('Throttle Integral * %.2f') % (throttle_to_STE_rate)])
        
        data_plot.add_graph(['total_energy_balance_rate',
                             'total_energy_balance_rate_sp',
                             'pitch_integ'],
                            colors8[3:6], 
                            ['Energy Balance Rate',
                             'Energy Balance Rate Setpoint',
                             'Energy Balance Rate (Pitch) Integral'],
                            mark_nan=True)
        
        
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
        if data_plot.finalize() is not None: plots.append(data_plot)
    
    except (KeyError, IndexError) as error:
        pass
    
    # manual control inputs
    # prefer the manual_control_setpoint topic. Old logs do not contain it
    if any(elem.name == 'manual_control_setpoint' for elem in data):
        data_plot = DataPlot(data, plot_config, 'manual_control_setpoint',
                             title='Manual Control Inputs (Radio or Joystick)',
                             plot_height='small', y_range=Range1d(-1.1, 1.1),
                             changed_params=changed_params, x_range=x_range)
        data_plot.add_graph(manual_control_sp_controls + ['aux1', 'aux2'], colors8[0:6],
                            ['Y / Roll', 'X / Pitch', 'Yaw',
                             'Throttle ' + manual_control_sp_throttle_range, 'Aux1', 'Aux2'])
        data_plot.change_dataset(manual_control_switches_topic)
        data_plot.add_graph([lambda data: ('mode_slot', data['mode_slot']/6),
                             lambda data: ('kill_switch', data['kill_switch'] == 1)],
                            colors8[6:8], ['Flight Mode', 'Kill Switch'])
        # TODO: add RTL switch and others? Look at params which functions are mapped?
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

        if data_plot.finalize() is not None: plots.append(data_plot)

    else: # it's an old log (COMPATIBILITY)
        data_plot = DataPlot(data, plot_config, 'rc_channels',
                             title='Raw Radio Control Inputs',
                             plot_height='small', y_range=Range1d(-1.1, 1.1),
                             changed_params=changed_params, x_range=x_range)
        num_rc_channels = 8
        if data_plot.dataset:
            max_channels = np.amax(data_plot.dataset.data['channel_count'])
            if max_channels < num_rc_channels: num_rc_channels = max_channels
        legends = []
        for i in range(num_rc_channels):
            channel_names = px4_ulog.get_configured_rc_input_names(i)
            if channel_names is None:
                legends.append('Channel '+str(i))
            else:
                legends.append('Channel '+str(i)+' ('+', '.join(channel_names)+')')
        data_plot.add_graph(['channels['+str(i)+']' for i in range(num_rc_channels)],
                            colors8[0:num_rc_channels], legends, mark_nan=True)
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

        if data_plot.finalize() is not None: plots.append(data_plot)

    # Delay between manual control inputs
    try:
        data_plot = DataPlot(data, plot_config, 'manual_control_setpoint',
                             title='Delay Between Manual Control Inputs',
                             plot_height='small',
                             changed_params=changed_params, x_range=x_range, y_axis_label = '[ms]')
        throttle_timestamps_us = ulog.get_dataset('manual_control_setpoint').data['timestamp_sample']
        data_plot.add_circle([lambda data: ('timestamp_sample', np.diff(throttle_timestamps_us)/1e3)], colors8[0:1],
                            ['Input Delay'])
    except (KeyError, IndexError) as error:
        print('throttle_timestamps_us does not exist: '+str(error))
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

    if data_plot.finalize() is not None: plots.append(data_plot)
    
    # Ping status
    data_plot = DataPlot(data, plot_config, 'ping',
                         y_start=0, title='Ping Status',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_circle(['rtt_ms'],
                        colors8[0:1], ['Round-trip Time [ms]'])
    data_plot.add_graph(['dropped_packets'],
                        colors8[1:2], ['# of dropped packets'], mark_nan=True)
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    # Radio status
    data_plot = DataPlot(data, plot_config, 'radio_status',
                         y_start=0, title='Radio Status',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range, y_axis_label = '[dBm]')
    data_plot.add_graph([lambda data: ('rssi_dbm', data['rssi_dbm'] - data['noise_dbm']),
                         'rssi_dbm', 'noise_dbm', 'rssi_antenna[0]', 'rssi_antenna[1]',
                         'mcs'],
                        colors8[0:6], ['SNR', 'Signal', 'Noise', 'ant0 signal', 'ant1 signal',
                                       'MCS'], mark_nan=True)
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    # Radio TX status
    data_plot = DataPlot(data, plot_config, 'radio_status',
                         y_start=0, title='Radio TX Status',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['tx_bytes', 'tx_retries', 'tx_failed'],
                        colors8[0:6], ['Bytes', 'Retries', 'Failed'], mark_nan=True)
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    
    # actuator controls 0
    data_plot = DataPlot(data, plot_config, actuator_controls_0.torque_sp_topic,
                         y_start=0, title='Actuator Controls',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(actuator_controls_0.torque_axes_field_names,
                        colors8[0:3], ['Roll', 'Pitch', 'Yaw'], mark_nan=True)
    data_plot.change_dataset(actuator_controls_0.thrust_sp_topic)
    if actuator_controls_0.thrust_z_neg is not None:
        data_plot.add_graph([lambda data: ('thrust', actuator_controls_0.thrust_z_neg)],
                            colors8[3:4], ['Thrust (up)'], mark_nan=True)
    if actuator_controls_0.thrust_x is not None:
        data_plot.add_graph([lambda data: ('thrust', actuator_controls_0.thrust_x)],
                            colors8[4:5], ['Thrust (forward)'], mark_nan=True)
    data_plot.change_dataset('actuator_controls_0')
    data_plot.add_graph(['control[4]', 'control[5]', 'control[6]'],
                        colors8[5:8], ['Flaps', 'Spoilers', 'Brakes'], mark_nan=True)
    data_plot.change_dataset('flaps_setpoint')
    data_plot.add_graph(['normalized_setpoint'],
                        colors8[5:6], ['Flaps'], mark_nan=True)
    data_plot.change_dataset('spoilers_setpoint')
    data_plot.add_graph(['normalized_setpoint'],
                        colors8[6:7], ['Spoilers'], mark_nan=True)
    data_plot.change_dataset('landing_gear_wheel')
    data_plot.add_graph(['normalized_wheel_setpoint'],
                        colors8[7:8], ['Steering'], mark_nan=True)
    data_plot.change_dataset('brakes_setpoint')
    data_plot.add_graph(['normalized_setpoint'],
                        colors13[8:9], ['Brakes'], mark_nan=True)
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)

    # actuator controls (Main) FFT (for filter & output noise analysis)
    data_plot = DataPlotFFT(data, plot_config, actuator_controls_0.torque_sp_topic,
                            title='Actuator Controls FFT', y_range = Range1d(0, 0.01))
    data_plot.add_graph(actuator_controls_0.torque_axes_field_names,
                        colors3, ['Roll', 'Pitch', 'Yaw'])
    if not data_plot.had_error:
        if 'MC_DTERM_CUTOFF' in ulog.initial_parameters: # COMPATIBILITY
            data_plot.mark_frequency(
                ulog.initial_parameters['MC_DTERM_CUTOFF'],
                'MC_DTERM_CUTOFF')
        if 'IMU_DGYRO_CUTOFF' in ulog.initial_parameters:
            data_plot.mark_frequency(
                ulog.initial_parameters['IMU_DGYRO_CUTOFF'],
                'IMU_DGYRO_CUTOFF')
        if 'IMU_GYRO_CUTOFF' in ulog.initial_parameters:
            data_plot.mark_frequency(
                ulog.initial_parameters['IMU_GYRO_CUTOFF'],
                'IMU_GYRO_CUTOFF', 20)

    data_plot.change_dataset(actuator_controls_0.thrust_sp_topic)
    data_plot.add_graph(actuator_controls_0.thrust_x,
                        color_gray, ['Throttle'])

    if data_plot.finalize() is not None: plots.append(data_plot)
    
    
    # angular_velocity FFT (for filter & output noise analysis)
    data_plot = DataPlotFFT(data, plot_config, 'vehicle_angular_velocity',
                            title='Angular Velocity FFT', y_range = Range1d(0, 0.03))
    data_plot.add_graph(['xyz[0]', 'xyz[1]', 'xyz[2]'],
                        colors3, ['Rollspeed', 'Pitchspeed', 'Yawspeed'])
    if not data_plot.had_error:
        if 'IMU_GYRO_CUTOFF' in ulog.initial_parameters:
            data_plot.mark_frequency(
                ulog.initial_parameters['IMU_GYRO_CUTOFF'],
                'IMU_GYRO_CUTOFF', 20)
        if 'IMU_GYRO_NF_FREQ' in ulog.initial_parameters:
            if  ulog.initial_parameters['IMU_GYRO_NF_FREQ'] > 0:
                data_plot.mark_frequency(
                    ulog.initial_parameters['IMU_GYRO_NF_FREQ'],
                    'IMU_GYRO_NF_FREQ', 70)

    if data_plot.finalize() is not None: plots.append(data_plot)


    # angular_acceleration FFT (for filter & output noise analysis)
    data_plot = DataPlotFFT(data, plot_config, 'vehicle_angular_acceleration',
                            title='Angular Acceleration FFT')
    data_plot.add_graph(['xyz[0]', 'xyz[1]', 'xyz[2]'],
                        colors3, ['Roll accel', 'Pitch accel', 'Yaw accel'])
    if not data_plot.had_error:
        if 'IMU_DGYRO_CUTOFF' in ulog.initial_parameters:
            data_plot.mark_frequency(
                ulog.initial_parameters['IMU_DGYRO_CUTOFF'],
                'IMU_DGYRO_CUTOFF')
        if 'IMU_GYRO_NF_FREQ' in ulog.initial_parameters:
            if  ulog.initial_parameters['IMU_GYRO_NF_FREQ'] > 0:
                data_plot.mark_frequency(
                    ulog.initial_parameters['IMU_GYRO_NF_FREQ'],
                    'IMU_GYRO_NF_FREQ', 70)

    if data_plot.finalize() is not None: plots.append(data_plot)

    # actuator controls 1 (torque + thrust)
    # (only present on VTOL, Fixed-wing config)
    data_plot = DataPlot(data, plot_config, actuator_controls_1.torque_sp_topic,
                         y_start=0, title='Actuator Controls 1 (VTOL in Fixed-Wing mode)',
                         plot_height='small', changed_params=changed_params, topic_instance=1,
                         x_range=x_range)
    data_plot.add_graph(actuator_controls_1.torque_axes_field_names,
                        colors8[0:3], ['Roll', 'Pitch', 'Yaw'], mark_nan=True)
    data_plot.change_dataset(actuator_controls_1.thrust_sp_topic,
                             actuator_controls_1.topic_instance)
    if actuator_controls_1.thrust_x is not None:
        data_plot.add_graph([lambda data: ('thrust', actuator_controls_1.thrust_x)],
                            colors8[3:4], ['Thrust (forward)'], mark_nan=True)
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)

    if dynamic_control_alloc:

        # actuator motors, actuator servos
        actuator_output_plots = [("actuator_motors", "Motor"), ("actuator_servos", "Servo")]
        for topic_name, plot_name in actuator_output_plots:

            data_plot = DataPlot(data, plot_config, topic_name,
                                 y_range=Range1d(-1, 1), title=plot_name+' Outputs',
                                 plot_height='small', changed_params=changed_params,
                                 x_range=x_range)
            num_actuator_outputs = 12
            if data_plot.dataset:
                for i in range(num_actuator_outputs):
                    try:
                        output_data = data_plot.dataset.data['control['+str(i)+']']
                    except KeyError:
                        num_actuator_outputs = i
                        break

                    if np.isnan(output_data).all():
                        num_actuator_outputs = i
                        break

                if num_actuator_outputs > 0:
                    data_plot.add_graph(['control['+str(i)+']'
                                         for i in range(num_actuator_outputs)],
                                        [colors8[i % 8] for i in range(num_actuator_outputs)],
                                        [plot_name+' '+str(i+1)
                                         for i in range(num_actuator_outputs)])
                    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
                    if data_plot.finalize() is not None: plots.append(data_plot)

    else:

        actuator_output_plots = [(0, "Actuator Outputs (Main)"), (1, "Actuator Outputs (AUX)"),
                                 (2, "Actuator Outputs (EXTRA)")]
        for topic_instance, plot_name in actuator_output_plots:

            data_plot = DataPlot(data, plot_config, 'actuator_outputs',
                                 y_start=0, title=plot_name, plot_height='small',
                                 changed_params=changed_params, topic_instance=topic_instance,
                                 x_range=x_range)
            num_actuator_outputs = 16
            # only plot if at least one of the outputs is not constant
            all_constant = True
            if data_plot.dataset:
                max_outputs = np.amax(data_plot.dataset.data['noutputs'])
                if max_outputs < num_actuator_outputs: num_actuator_outputs = max_outputs

                for i in range(num_actuator_outputs):
                    output_data = data_plot.dataset.data['output['+str(i)+']']
                    if not np.all(output_data == output_data[0]):
                        all_constant = False

            if not all_constant:
                data_plot.add_graph(['output['+str(i)+']' for i in range(num_actuator_outputs)],
                                    [colors8[i % 8] for i in range(num_actuator_outputs)],
                                    ['Output '+str(i) for i in range(num_actuator_outputs)],
                                    mark_nan=True)
                plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

                if data_plot.finalize() is not None: plots.append(data_plot)

    # Engine log
    data_plot = DataPlot(data, plot_config, 'kingtech_g4p_report',
                         y_axis_label='', title='Engine Log',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph([lambda data: ('turbine_rpm', data['turbine_rpm']/1000), 'turbine_pw_raw',
                         'turbine_temperature_raw', 'turbine_power_bar_percent'], 
                        colors8[0:4], ['RPM/1000', 'Pump Power', 'Temperature [deg C]', 'Turbine Power'])
    
    data_plot.change_dataset(actuator_controls_0.torque_sp_topic)
    data_plot.add_graph([lambda data: ('thrust', actuator_controls_0.thrust_x * 100.0)],
                        colors8[4:5], ['Throttle [%]'], mark_nan=True)

    poly_fit_g_per_min = [3.02999456e-02, -2.98917128e-01,  1.17480328e+02 ] 

    diesel_no2_kgPerL = 0.85
    data_plot.add_graph([lambda data: ('thrust', np.polyval(poly_fit_g_per_min, actuator_controls_0.thrust_x * 100.0)/60.0 / diesel_no2_kgPerL)],
                        colors8[5:6], ['Est. Fuel Consumption [mL/s]'], mark_nan=True)
    throttle_timestamp = ulog.get_dataset(actuator_controls_0.torque_sp_topic).data['timestamp']
    throttle = actuator_controls_0.thrust_x * 100.0

    data_plot.add_graph([lambda data: ('thrust', integrate.cumtrapz(
        np.polyval(poly_fit_g_per_min, throttle),
        throttle_timestamp * 1e-6) /1000.0/60.0/ diesel_no2_kgPerL)],
                        colors8[6:7], ['Est. Fuel Consumed [L]'], mark_nan=True)
    
    data_plot.change_dataset('fuel_tank_status')
    data_plot.add_graph([lambda data: ('maximum_fuel_capacity', data['maximum_fuel_capacity']/1e3),
                         lambda data: ('consumed_fuel', data['consumed_fuel']/1e3), 
                         'fuel_consumption_rate'],
                        colors13[7:10], 
                        ['Logged Max Fuel Capacity [L]', 'Logged Consumed Fuel [L]', 'Logged Fuel Consumption Rate [mL/s]'],
                        mark_nan=True)
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    
    # raw acceleration
    data_plot = DataPlot(data, plot_config, 'sensor_combined',
                         y_axis_label='[m/s^2]', title='Raw Acceleration',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['accelerometer_m_s2[0]', 'accelerometer_m_s2[1]',
                         'accelerometer_m_s2[2]'], colors3, ['X', 'Y', 'Z'])
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)

    # Vibration Metrics
    data_plot = DataPlot(data, plot_config, 'vehicle_imu_status',
                         title='Vibration Metrics',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range, y_start=0, topic_instance=0)
    data_plot.add_graph(['accel_vibration_metric'], colors8[0:1],
                         ['Accel 0 Vibration Level [m/s^2]'])

    data_plot.change_dataset('vehicle_imu_status', 1)
    data_plot.add_graph(['accel_vibration_metric'], colors8[1:2],
                            ['Accel 1 Vibration Level [m/s^2]'])

    data_plot.change_dataset('vehicle_imu_status', 2)
    data_plot.add_graph(['accel_vibration_metric'], colors8[2:3],
                            ['Accel 2 Vibration Level [m/s^2]'])

    data_plot.change_dataset('vehicle_imu_status', 3)
    data_plot.add_graph(['accel_vibration_metric'], colors8[3:4],
                            ['Accel 3 Vibration Level [rad/s]'])

    data_plot.add_horizontal_background_boxes(
        ['green', 'orange', 'red'], [4.905, 9.81])

    if data_plot.finalize() is not None: plots.append(data_plot)

    # Acceleration Spectrogram
    data_plot = DataPlotSpec(data, plot_config, 'sensor_combined',
                             y_axis_label='[Hz]', title='Acceleration Power Spectral Density',
                             plot_height='small', x_range=x_range)
    data_plot.add_graph(['accelerometer_m_s2[0]', 'accelerometer_m_s2[1]', 'accelerometer_m_s2[2]'],
                        ['X', 'Y', 'Z'])
    if data_plot.finalize() is not None: plots.append(data_plot)


    # Filtered Gyro (angular velocity) Spectrogram
    data_plot = DataPlotSpec(data, plot_config, 'vehicle_angular_velocity',
                             y_axis_label='[Hz]', title='Angular velocity Power Spectral Density',
                             plot_height='small', x_range=x_range)
    data_plot.add_graph(['xyz[0]', 'xyz[1]', 'xyz[2]'],
                        ['rollspeed', 'pitchspeed', 'yawspeed'])

    if data_plot.finalize() is not None: plots.append(data_plot)


    # Filtered angular acceleration Spectrogram
    data_plot = DataPlotSpec(data, plot_config, 'vehicle_angular_acceleration',
                             y_axis_label='[Hz]',
                             title='Angular acceleration Power Spectral Density',
                             plot_height='small', x_range=x_range)
    data_plot.add_graph(['xyz[0]', 'xyz[1]', 'xyz[2]'],
                        ['roll accel', 'pitch accel', 'yaw accel'])

    if data_plot.finalize() is not None: plots.append(data_plot)


    # raw angular speed
    data_plot = DataPlot(data, plot_config, 'sensor_combined',
                         y_axis_label='[deg/s]', title='Raw Angular Speed (Gyroscope)',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph([
        lambda data: ('gyro_rad[0]', np.rad2deg(data['gyro_rad[0]'])),
        lambda data: ('gyro_rad[1]', np.rad2deg(data['gyro_rad[1]'])),
        lambda data: ('gyro_rad[2]', np.rad2deg(data['gyro_rad[2]']))],
                        colors3, ['X', 'Y', 'Z'])
    if data_plot.finalize() is not None: plots.append(data_plot)

    # FIFO accel
    if add_virtual_fifo_topic_data(ulog, 'sensor_accel_fifo'):
        # Raw data
        data_plot = DataPlot(data, plot_config, 'sensor_accel_fifo_virtual',
                             y_axis_label='[m/s^2]', title='Raw Acceleration (FIFO)',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        data_plot.add_graph(['x', 'y', 'z'], colors3, ['X', 'Y', 'Z'])
        if data_plot.finalize() is not None: plots.append(data_plot)

        # power spectral density
        data_plot = DataPlotSpec(data, plot_config, 'sensor_accel_fifo_virtual',
                                 y_axis_label='[Hz]',
                                 title='Acceleration Power Spectral Density (FIFO)',
                                 plot_height='normal', x_range=x_range)
        data_plot.add_graph(['x', 'y', 'z'], ['X', 'Y', 'Z'])
        if data_plot.finalize() is not None: plots.append(data_plot)

        # sampling regularity
        data_plot = DataPlot(data, plot_config, 'sensor_accel_fifo', y_range=Range1d(0, 25e3),
                             y_axis_label='[us]',
                             title='Sampling Regularity of Sensor Data (FIFO)', plot_height='small',
                             changed_params=changed_params, x_range=x_range)
        sensor_accel_fifo = ulog.get_dataset('sensor_accel_fifo').data
        sampling_diff = np.diff(sensor_accel_fifo['timestamp'])
        min_sampling_diff = np.amin(sampling_diff)
        plot_dropouts(data_plot.bokeh_plot, ulog.dropouts, min_sampling_diff)
        data_plot.add_graph([lambda data: ('timediff', np.append(sampling_diff, 0))],
                            [colors3[2]], ['delta t (between 2 logged samples)'])
        if data_plot.finalize() is not None: plots.append(data_plot)

    # FIFO gyro
    if add_virtual_fifo_topic_data(ulog, 'sensor_gyro_fifo'):
        # Raw data
        data_plot = DataPlot(data, plot_config, 'sensor_gyro_fifo_virtual',
                             y_axis_label='[m/s^2]', title='Raw Gyro (FIFO)',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        data_plot.add_graph(['x', 'y', 'z'], colors3, ['X', 'Y', 'Z'])
        data_plot.add_graph([
            lambda data: ('x', np.rad2deg(data['x'])),
            lambda data: ('y', np.rad2deg(data['y'])),
            lambda data: ('z', np.rad2deg(data['z']))],
                            colors3, ['X', 'Y', 'Z'])
        if data_plot.finalize() is not None: plots.append(data_plot)

        # power spectral density
        data_plot = DataPlotSpec(data, plot_config, 'sensor_gyro_fifo_virtual',
                                 y_axis_label='[Hz]', title='Gyro Power Spectral Density (FIFO)',
                                 plot_height='normal', x_range=x_range)
        data_plot.add_graph(['x', 'y', 'z'], ['X', 'Y', 'Z'])
        if data_plot.finalize() is not None: plots.append(data_plot)


    # magnetic field strength
    data_plot = DataPlot(data, plot_config, 'sensor_mag',
                         y_axis_label='[gauss]', title='Raw Magnetic Field Strength',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range, topic_instance=0)
    
    for mag_i in range(3):
        data_plot.change_dataset('sensor_mag', mag_i)
        data_plot.add_graph(['x', 'y',
                             'z', lambda data: ('temperature', np.sqrt(data['x']**2 + data['y']**2 + data['z']**2))], 
                            colors13[0+(4*mag_i):4+(4*mag_i)],
                            ['X_'+str(mag_i), 'Y_'+str(mag_i), 'Z_'+str(mag_i), 'Abs_'+str(mag_i)])
    if data_plot.finalize() is not None: plots.append(data_plot)


    # distance sensor
    data_plot = DataPlot(data, plot_config, 'distance_sensor',
                         y_start=0, y_axis_label='[m]', title='Distance Sensor',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['current_distance', 'variance'], colors3[0:2],
                        ['Distance', 'Variance'])
    if data_plot.finalize() is not None: plots.append(data_plot)



    # gps uncertainty
    # the accuracy values can be really large if there is no fix, so we limit the
    # y axis range to some sane values
    data_plot = DataPlot(data, plot_config, 'vehicle_gps_position',
                         title='GPS Uncertainty', y_range=Range1d(0, 40),
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['eph', 'epv', 'satellites_used', 'fix_type'], colors8[::2],
                        ['Horizontal position accuracy [m]', 'Vertical position accuracy [m]',
                         'Num Satellites used', 'GPS Fix'])
    if data_plot.finalize() is not None: plots.append(data_plot)


    # gps noise & jamming
    data_plot = DataPlot(data, plot_config, 'vehicle_gps_position',
                         y_start=0, title='GPS Noise & Jamming',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['noise_per_ms', 'jamming_indicator'], colors3[0:2],
                        ['Noise per ms', 'Jamming Indicator'])
    if data_plot.finalize() is not None: plots.append(data_plot)

    # gps signal to noise ratio
    try:
        data_plot = DataPlot(data, plot_config, 'satellite_info',
                             y_start=0, title='GPS Satellite SNR',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        satellite_data      = ulog.get_dataset('satellite_info')
        sample_snr          = satellite_data.data['snr[0]']
        zero_snr            = np.zeros(np.shape(sample_snr))
        max_snr             = np.zeros(np.shape(sample_snr))
        min_snr             = 99 * np.ones(np.shape(sample_snr))
        mean_snr            = np.zeros(np.shape(sample_snr))
        num_sats_used       = np.zeros(np.shape(sample_snr))
        num_sats_above_th   = np.zeros(np.shape(sample_snr))
        snr_dict            = dict()
        num_sats = 20
        db_threshold = 30
        for i in range(0, num_sats):
            snr_name    = ('snr[%i]') % (i)
            used_name   = ('used[%i]') % (i)
            snr_data    = satellite_data.data[snr_name]
            used_data   = satellite_data.data[used_name]
            is_zero     = snr_data == 0
            is_not_zero = np.bitwise_not(is_zero)
            # is_above_th = snr_data > db_threshold
            
            gt_max      = snr_data > max_snr
            lt_min      = np.bitwise_and(snr_data < min_snr, is_not_zero)
            max_snr[gt_max] = snr_data[gt_max]
            min_snr[lt_min] = snr_data[lt_min]
            
            zero_snr += snr_data == 0
            mean_snr += snr_data
            num_sats_used += used_data
            num_sats_above_th += snr_data > db_threshold
            
        mean_snr = mean_snr / (num_sats - zero_snr)
        num_pos_snr = num_sats - zero_snr
        data_plot.add_graph([lambda data: (snr_name, max_snr)], colors3[0:1],
                            [('Max SNR; avg = %.1f, stdev = %.1f') % (np.mean(max_snr), np.std(max_snr))])
        data_plot.add_graph([lambda data: (snr_name, min_snr)], colors3[1:2],
                        [('Min SNR; avg = %.1f, stdev = %.1f') % (np.mean(min_snr), np.std(min_snr))])
        data_plot.add_graph([lambda data: (snr_name, mean_snr)], colors3[2:3],
                    [('Mean SNR; avg = %.1f, stdev = %.1f') % (np.mean(mean_snr), np.std(mean_snr))])
        data_plot.add_graph([lambda data: (snr_name, num_pos_snr)], colors8[3:4],
                    [('# of positive SNR; avg = %.1f, stdev = %.1f') % (np.mean(num_pos_snr), np.std(num_pos_snr))])
        data_plot.add_graph([lambda data: (snr_name, num_sats_used)], colors8[4:5],
                    [('# of satellites used; avg = %.1f, stdev = %.1f') % (np.mean(num_sats_used), np.std(num_sats_used))])
        data_plot.add_graph([lambda data: (snr_name, num_sats_above_th)], colors8[5:6],
                    [('# of satellites above %i [dBHz]; avg = %.2f, stdev = %.1f') % (db_threshold, np.mean(num_sats_above_th), np.std(num_sats_above_th))])
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
        if data_plot.finalize() is not None: plots.append(data_plot)
    
        snr_dict['max'] = max_snr
        snr_dict['min'] = min_snr
        snr_dict['mean'] = mean_snr
        snr_dict['num_sats_above_th'] = num_sats_above_th
        snr_dict['num_sats_used'] = num_sats_used
        snr_dict['timestamp'] = satellite_data.data['timestamp']
    
    except:
        pass

    # thrust and magnetic field
    data_plot = DataPlot(data, plot_config, magnetometer_ga_topic,
                         y_start=0, title='Thrust and Magnetic Field', plot_height='small',
                         changed_params=changed_params, x_range=x_range, topic_instance=0)
    data_plot.add_graph(
        [lambda data: ('len_mag', np.sqrt(data['magnetometer_ga[0]']**2 +
                                          data['magnetometer_ga[1]']**2 +
                                          data['magnetometer_ga[2]']**2))],
        colors3[0:1], ['Norm of Magnetic Field'])
    data_plot.change_dataset(actuator_controls_0.thrust_sp_topic)
    if actuator_controls_0.thrust is not None:
        data_plot.add_graph([lambda data: ('thrust', actuator_controls_0.thrust)],
                            colors3[1:2], ['Thrust'])
    if is_vtol and not dynamic_control_alloc:
        data_plot.change_dataset(actuator_controls_1.thrust_sp_topic)
        if actuator_controls_1.thrust_x is not None:
            data_plot.add_graph([lambda data: ('thrust', actuator_controls_1.thrust_x)],
                                colors3[2:3], ['Thrust (Fixed-wing'])
    if data_plot.finalize() is not None: plots.append(data_plot)



    # Estimate the gap between the magnetic heading and the yaw estimate
    heading_gap = 0
    heading_rms = 0
    declination_deg = ulog.initial_parameters['EKF2_MAG_DECL']
    try:
        yaw_set = ulog.get_dataset('vehicle_attitude')
        yaw_time = yaw_set.data['timestamp']
        yaw_data = np.rad2deg(yaw_set.data['yaw'])
        
        mag_set = ulog.get_dataset(magnetometer_ga_topic)
        mag_time = mag_set.data['timestamp']
        mag_x = mag_set.data['magnetometer_ga[0]']
        mag_y = mag_set.data['magnetometer_ga[1]']
        mag_z = mag_set.data['magnetometer_ga[2]']
        
        att_data = ulog.get_dataset('vehicle_attitude')
        phi     = np.interp(mag_time, att_data.data['timestamp'], att_data.data['roll'])
        theta   = np.interp(mag_time, att_data.data['timestamp'], att_data.data['pitch'])
        psi     = 0#np.interp(mag_time, att_data.data['timestamp'], att_data.data['yaw'])
        
        mag_earth_pred_x = mag_z * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) - mag_y * (np.cos(phi) * np.sin(psi) - np.cos(psi) * np.sin(phi) * np.sin(theta)) + mag_x * np.cos(psi) * np.cos(theta)
        mag_earth_pred_y = mag_y * (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(psi) * np.sin(theta)) - mag_z * (np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi) * np.sin(theta)) + mag_x * np.cos(theta) * np.sin(psi)
        mag_earth_pred_z = mag_z * np.cos(phi) * np.cos(theta) - mag_x * np.sin(theta) + mag_y * np.cos(theta) * np.sin(phi)
        
        mag_heading  = -np.rad2deg(np.arctan2(mag_earth_pred_y, mag_earth_pred_x)) + declination_deg#-np.rad2deg(np.arctan2(mag_y, mag_x)) + declination_deg
        
        mag_heading_uw = mag_heading # magnetic heading without wrapping around -180 and 180 [deg]
        yaw_uw = yaw_data # estimated heading without wrapping around -180 and 180 [deg]
        
        for idx in range(1, len(mag_heading_uw)):
            if mag_heading_uw[idx] - mag_heading_uw[idx-1] > 200:
                mag_heading_uw[idx] -= 360
            elif mag_heading_uw[idx] - mag_heading_uw[idx-1] < -200:
                mag_heading_uw[idx] += 360
                
        for idx in range(1, len(yaw_uw)):
            if yaw_uw[idx] - yaw_uw[idx-1] > 200:
                yaw_uw[idx] -= 360
            elif yaw_uw[idx] - yaw_uw[idx-1] < -200:
                yaw_uw[idx] += 360
        
        (heading_gap, heading_rms, heading_diff) = get_mean_gap(yaw_time, yaw_uw, mag_time, mag_heading_uw)
    
        psi     = np.interp(mag_time, att_data.data['timestamp'], att_data.data['yaw'])
        
        mag_earth_pred_x = mag_z * (np.sin(phi) * np.sin(psi) + np.cos(phi) * np.cos(psi) * np.sin(theta)) - mag_y * (np.cos(phi) * np.sin(psi) - np.cos(psi) * np.sin(phi) * np.sin(theta)) + mag_x * np.cos(psi) * np.cos(theta)
        mag_earth_pred_y = mag_y * (np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(psi) * np.sin(theta)) - mag_z * (np.cos(psi) * np.sin(phi) - np.cos(phi) * np.sin(psi) * np.sin(theta)) + mag_x * np.cos(theta) * np.sin(psi)
        mag_earth_pred_z = mag_z * np.cos(phi) * np.cos(theta) - mag_x * np.sin(theta) + mag_y * np.cos(theta) * np.sin(phi)
    
        # Earth magnetic field NED and Body magnetic field
        data_plot = DataPlot(data, plot_config, 'estimator_states',
                             y_axis_label='[G]', title='Earth Magnetic Field Estimate and Body Magnetic Field Correction',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        data_plot.add_graph(['states[16]', 'states[17]', 'states[18]', 'states[19]', 'states[20]', 'states[21]'],
                                    colors8[0:6], ['Earth N', 'Earth E', 'Earth D', 'Body X corr', 'Body Y corr', 'Body Z corr'])
        
        data_plot.change_dataset(magnetometer_ga_topic)
        data_plot.add_graph(
            [lambda data: ('len_mag', mag_earth_pred_x)],
            colors8[6:7], ['Earth N (Est)'])
        data_plot.add_graph(
            [lambda data: ('len_mag', mag_earth_pred_y)],
            colors8[7:8], ['Earth E (Est)'])
        data_plot.add_graph(
            [lambda data: ('len_mag', mag_earth_pred_z)],
            colors8[0:1], ['Earth D (Est)'])
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
        if data_plot.finalize() is not None: plots.append(data_plot)
    
        # Magnetic heading
    
        data_plot = DataPlot(data, plot_config, magnetometer_ga_topic,
                             y_axis_label='[deg]', title='Compass heading',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        data_plot.add_graph(
            [lambda data: ('len_mag', mag_heading_uw)],
            colors8[0:1], [('Magnetic Heading + %.2f [deg] declination') % (declination_deg)])
        #Phi
        data_plot.add_graph(
            [lambda data: ('len_mag', np.rad2deg(np.arctan(np.sqrt((data['magnetometer_ga[0]']**2+data['magnetometer_ga[1]']**2)/ data['magnetometer_ga[2]']))))],
            colors8[1:2], ['Magnetic Dip'])
    
        data_plot.change_dataset('vehicle_attitude')
        data_plot.add_graph([lambda data: ('yaw', yaw_uw)],
                            colors8[2:3], ['Yaw Estimated'], mark_nan=True)
        data_plot.add_graph([lambda data: ('yaw', -heading_diff)],
                            colors8[4:5], [('Yaw - Mag Heading: mean=%.2f; RMS=%.2f') % (-heading_gap, heading_rms)], mark_nan=True)
        if data_plot.finalize() is not None: plots.append(data_plot)
        
    except (KeyError, IndexError) as error:
        print('Could not compute yaw gap')
        
        
    # power
    data_plot = DataPlot(data, plot_config, 'battery_status',
                         y_start=0, title='Power',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['voltage_v', 'voltage_filtered_v',
                         'current_a', lambda data: ('discharged_mah', data['discharged_mah']/100),
                         lambda data: ('remaining', data['remaining']*10)],
                        colors8[::2]+colors8[1:2],
                        ['Battery Voltage [V]', 'Battery Voltage filtered [V]',
                         'Battery Current [A]', 'Discharged Amount [mAh / 100]',
                         'Battery remaining [0=empty, 10=full]'])
    data_plot.change_dataset('system_power')
    if data_plot.dataset:
        if 'voltage5v_v' in data_plot.dataset.data and \
                        np.amax(data_plot.dataset.data['voltage5v_v']) > 0.0001:
            data_plot.add_graph(['voltage5v_v'], colors8[7:8], ['5 V'])
        if 'sensors3v3[0]' in data_plot.dataset.data and \
                        np.amax(data_plot.dataset.data['sensors3v3[0]']) > 0.0001:
            data_plot.add_graph(['sensors3v3[0]'], colors8[5:6], ['3.3 V'])
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    
    # Power monitors
    data_plot = DataPlot(data, plot_config, 'power_monitor',
                         y_start=0, title='Power Monitors',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    for pm_i in range(3):
        data_plot.change_dataset('power_monitor', pm_i)
        data_plot.add_graph(['voltage_v', 'current_a'],
                            colors13[3*pm_i:3*pm_i+2],
                            [('V_%i') % (pm_i), ('Amp_%i') % (pm_i)])
    
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    # CANBUS errors
    data_plot = DataPlot(data, plot_config, 'uavcan_status',
                          y_start=0, title='UAVCAN status',
                          plot_height='small', changed_params=changed_params,
                          x_range=x_range)

    data_plot.add_graph(['transfer_errors', 'internal_failures', 'rx_transfers', 'tx_transfers'], colors3[0:4],
                        ['Transfer Errors', 'Internal Failures', 'RX Transfers', 'TX Transfers'])
    for can_i in range(2):
        data_plot.add_graph([('can_hw_errors[%i]') % (can_i), ('can_io_errors[%i]') % (can_i),
                             ('can_rx_frames[%i]') % (can_i)],
                            colors13[3*can_i+4:3*can_i+7],
                            [('hw_errors_%i') % (can_i), ('io_errors_%i') % (can_i),
                             ('rx_frames_%i') % (can_i)])
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    
    # CANBUS errors time rate of change
    try:
        data_plot = DataPlot(data, plot_config, 'uavcan_status',
                              y_start=0, title='UAVCAN status time rate of change',
                              plot_height='small', changed_params=changed_params,
                              x_range=x_range, y_axis_label = '[#/s]')
        
        uavcan_status       = ulog.get_dataset('uavcan_status')
        timestamps          = uavcan_status.data['timestamp']
        transfer_errors     = uavcan_status.data['transfer_errors']
        internal_failures   = uavcan_status.data['internal_failures']
        rx_transfers        = uavcan_status.data['rx_transfers']
        tx_transfers        = uavcan_status.data['tx_transfers']
        
        data_plot.add_graph([lambda data: ('transfer_errors',   1e6 * np.append(np.diff(transfer_errors)/np.diff(timestamps), 0)),
                             lambda data: ('internal_failures', 1e6 * np.append(np.diff(internal_failures)/np.diff(timestamps), 0)),
                             lambda data: ('rx_transfers',      1e6 * np.append(np.diff(rx_transfers)/np.diff(timestamps), 0)),
                             lambda data: ('tx_transfers',      1e6 * np.append(np.diff(tx_transfers)/np.diff(timestamps), 0))                         ],
                                colors3[0:4], ['Transfer Errors', 'Internal Failures', 'RX Transfers', 'TX Transfers'])
        
        for can_i in range(2):
            hw_errors_i = uavcan_status.data[('can_hw_errors[%i]') % (can_i)]
            io_errors_i = uavcan_status.data[('can_io_errors[%i]') % (can_i)]
            rx_frames_i = uavcan_status.data[('can_rx_frames[%i]') % (can_i)]
            
            data_plot.add_graph([lambda data: (('can_hw_errors[%i]') % (can_i), 1e6 * np.append(np.diff(hw_errors_i)/np.diff(timestamps), 0)),
                                 lambda data: (('can_io_errors[%i]') % (can_i), 1e6 * np.append(np.diff(io_errors_i)/np.diff(timestamps), 0)),
                                 lambda data: (('can_rx_frames[%i]') % (can_i), 1e6 * np.append(np.diff(rx_frames_i)/np.diff(timestamps), 0)) ],
                                    colors13[3*can_i+4:3*can_i+7], 
                                    [('hw_errors_%i') % (can_i), ('io_errors_%i') % (can_i),
                                      ('rx_frames_%i') % (can_i)])
        
        plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
        if data_plot.finalize() is not None: plots.append(data_plot)
        
    except (KeyError, IndexError) as error:
            print('Error in estimator plot: '+str(error))
    


    #Temperature
    data_plot = DataPlot(data, plot_config, 'sensor_baro',
                         y_start=0, y_axis_label='[C]', title='Temperature',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['temperature'], colors8[0:1],
                        ['Baro temperature'])
    data_plot.change_dataset('sensor_accel')
    data_plot.add_graph(['temperature'], colors8[2:3],
                        ['Accel temperature'])
    data_plot.change_dataset('airspeed')
    data_plot.add_graph(['air_temperature_celsius'], colors8[4:5],
                        ['Airspeed temperature'])
    data_plot.change_dataset('battery_status')
    data_plot.add_graph(['temperature'], colors8[6:7],
                        ['Battery temperature'])
    if data_plot.finalize() is not None: plots.append(data_plot)


    # estimator flags
    try:
        data_plot = DataPlot(data, plot_config, 'estimator_status',
                             y_start=0, title='Estimator Flags',
                             plot_height='small', changed_params=changed_params,
                             x_range=x_range)
        estimator_status = ulog.get_dataset('estimator_status').data
        plot_data = []
        plot_labels = []
        input_data = [
            ('Health Flags (vel, pos, hgt)', estimator_status['health_flags']),
            ('Timeout Flags (vel, pos, hgt)', estimator_status['timeout_flags']),
            ('Velocity Check Bit', (estimator_status['innovation_check_flags'])&0x1),
            ('Horizontal Position Check Bit', (estimator_status['innovation_check_flags']>>1)&1),
            ('Vertical Position Check Bit', (estimator_status['innovation_check_flags']>>2)&1),
            ('Mag X, Y, Z Check Bits', (estimator_status['innovation_check_flags']>>3)&0x7),
            ('Yaw Check Bit', (estimator_status['innovation_check_flags']>>6)&1),
            ('Airspeed Check Bit', (estimator_status['innovation_check_flags']>>7)&1),
            ('Synthetic Sideslip Check Bit', (estimator_status['innovation_check_flags']>>8)&1),
            ('Height to Ground Check Bit', (estimator_status['innovation_check_flags']>>9)&1),
            ('Optical Flow X, Y Check Bits', (estimator_status['innovation_check_flags']>>10)&0x3),
            ]
        # filter: show only the flags that have non-zero samples
        for cur_label, cur_data in input_data:
            if np.amax(cur_data) > 0.1:
                data_label = 'flags_'+str(len(plot_data)) # just some unique string
                plot_data.append(lambda d, data=cur_data, label=data_label: (label, data))
                plot_labels.append(cur_label)
                if len(plot_data) >= 8: # cannot add more than that
                    break

        if len(plot_data) == 0:
            # add the plot even in the absence of any problem, so that the user
            # can validate that (otherwise it's ambiguous: it could be that the
            # estimator_status topic is not logged)
            plot_data = [lambda d: ('flags', input_data[0][1])]
            plot_labels = [input_data[0][0]]
        data_plot.add_graph(plot_data, colors8[0:len(plot_data)], plot_labels)
        if data_plot.finalize() is not None: plots.append(data_plot)
    except (KeyError, IndexError) as error:
        print('Error in estimator plot: '+str(error))

    # Test Ratios to identify faulty estimates
    data_plot = DataPlot(data, plot_config, 'estimator_status',
                         y_axis_label='', title='Estimator Status test ratios',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['pos_horiz_accuracy', 'pos_vert_accuracy', 'mag_test_ratio', 'vel_test_ratio', 
                             'pos_test_ratio', 'hgt_test_ratio', 'tas_test_ratio', 'hagl_test_ratio', 'beta_test_ratio'],
                                colors13[0:9], ['pos horiz accuracy', 'pos vert accuracy', 'mag', 'vel', 'pos', 'hgt', 'tas', 'hagl', 'beta'])
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    # Test Ratios to identify faulty sensor readings
    data_plot = DataPlot(data, plot_config, 'estimator_innovation_test_ratios',
                         y_axis_label='', title='Estimator Status innovation test ratios',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    data_plot.add_graph(['baro_vpos', 'rng_vpos', 'gps_hvel[0]', 'gps_hvel[1]', 'gps_vvel', 'gps_hpos[0]', 'gps_hpos[1]', 'heading'],
                                colors8[0:8], ['barometer', 'range sensor', 'gps hvel 0', 'gps hvel 1', 'gps vert vel', 'gps hpos 0', 'gps hpos 1', 'heading'])
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    # Height Fusion Diagnostics
    data_plot = DataPlot(data, plot_config, 'estimator_innovations',
                         y_start=0, y_axis_label='[m]', title='Height Fusion Diagnostics',
                         plot_height='small', changed_params=changed_params,
                         x_range=x_range)
    
    data_plot.add_graph(['baro_vpos', 'rng_vpos', 'gps_vpos', 'gps_vvel'], colors8[0:4], 
                        ['barometer innovation', 'range sensor innovation', 'gps_vpos innovation', 'gps_vvel innovation'])
    
    data_plot.change_dataset('estimator_innovation_variances')
    data_plot.add_graph(['baro_vpos', 'rng_vpos', 'gps_vpos'], colors8[4:7], 
                        ['barometer var', 'range sensor var'])
    
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    
    # Barometer-IMU consistency
    try:
        accel = ulog.get_dataset('sensor_combined')
        accel_t = accel.data['timestamp']
        
        estimator  = ulog.get_dataset('estimator_states')
        x_bias = np.interp(accel_t, estimator.data['timestamp'], estimator.data['states[13]'])
        y_bias = np.interp(accel_t, estimator.data['timestamp'], estimator.data['states[14]'])
        z_bias = np.interp(accel_t, estimator.data['timestamp'], estimator.data['states[15]'])
        
        accel_x = accel.data['accelerometer_m_s2[0]'] - x_bias
        accel_y = accel.data['accelerometer_m_s2[1]'] - y_bias
        accel_z = accel.data['accelerometer_m_s2[2]'] - z_bias
        
        baro = ulog.get_dataset(baro_alt_meter_topic)
        baro_alt = baro.data['baro_alt_meter']
        baro_t   = baro.data['timestamp']
        
        sos = signal.butter(10, 3, 'low', fs = 15, output='sos')
        baro_filt = (signal.sosfilt(sos, np.flip( signal.sosfilt(sos, np.flip(baro_alt)))))
        
        baro_dzdt = np.diff(baro_filt)/np.diff(baro_t) * 1e6
        
        pos = ulog.get_dataset('vehicle_local_position')
        pos_t = pos.data['timestamp']
        pos_z = pos.data['z'] - pos.data['z'][0]
        vel_z = pos.data['vz']
        
        gps_vz = ulog.get_dataset(baro_alt_meter_topic)
        baro_alt = baro.data['baro_alt_meter']
        baro_t   = baro.data['timestamp']
    except:
        pass

    try:
        x_rot, y_rot, z_rot = rotate_FRD_to_NED(ulog, accel_t, accel_x, accel_y, accel_z)
        accel_vel = -integrate.cumtrapz(z_rot + 9.80665, accel_t * 1e-6, initial = 0)
        
        
        # Make sure the accelerometer integration "snaps" to the altitude estimate every once in a while to see the IMU bias
        snap_dt_us = 1e6 # time interval between "re-snapping"
        snap_t = np.arange(np.min(accel_t), np.max(accel_t), snap_dt_us)
        accel_vel_steps = np.interp(snap_t, accel_t, accel_vel)
        avs_func        = interpolate.interp1d(snap_t, accel_vel_steps, kind = 'previous', fill_value = "extrapolate")
        
        local_vel_steps = np.interp(snap_t, pos_t, vel_z)
        lvs_func        = interpolate.interp1d(snap_t, local_vel_steps, kind = 'previous', fill_value = "extrapolate")
        
        snapped_accel_vel = accel_vel - avs_func(accel_t) - lvs_func(accel_t)
        
        accel_pos = integrate.cumtrapz(snapped_accel_vel, accel_t * 1e-6, initial = 0)
        
        accel_pos_steps = np.interp(snap_t, accel_t, accel_pos)
        aps_func        = interpolate.interp1d(snap_t, accel_pos_steps, kind = 'previous', fill_value = "extrapolate")
        
        local_pos_steps = np.interp(snap_t, pos_t, pos_z)
        lps_func        = interpolate.interp1d(snap_t, local_pos_steps, kind = 'previous', fill_value = "extrapolate")
        
        snapped_accel_pos = accel_pos - aps_func(accel_t) - lps_func(accel_t)
        
        
        data_plot = DataPlot(data, plot_config, 'sensor_combined',
                              y_start=0, title='Barometer-IMU consistency',
                              plot_height='small', changed_params=changed_params,
                              x_range=x_range)
        
        data_plot.add_graph([lambda data: ('accelerometer_m_s2[2]', snapped_accel_pos)], colors8[0:1], 
                            ['Accel Z'])
        data_plot.change_dataset(baro_alt_meter_topic)
        data_plot.add_graph([lambda data: ('baro_alt_meter', offset_baro_alt + ekf_data.data['states[9]'][0])], colors8[1:2], 
                            ['Baro Z'])
        data_plot.change_dataset('estimator_states')
        data_plot.add_graph([lambda data: ('states[9]', -(ekf_data.data['states[9]'] - ekf_data.data['states[9]'][0]))], colors8[5:6], 
                            ['EKF Z'])
        data_plot.change_dataset('vehicle_local_position')
        data_plot.add_graph([lambda data: ('z', -(data['z'] - data['z'][0]))],
                              colors8[2:3], ['Local Position Z'])
        
        data_plot.change_dataset('sensor_combined')
        data_plot.add_graph([lambda data: ('accelerometer_m_s2[2]', snapped_accel_vel)], colors8[3:4], 
                            ['Accel VZ'])
        data_plot.change_dataset(baro_alt_meter_topic)
        data_plot.add_graph([lambda data: ('baro_alt_meter', baro_dzdt)], colors8[1:2], 
                            ['Baro VZ'])
        data_plot.change_dataset('vehicle_local_position')
        data_plot.add_graph([lambda data: ('vz', -data['vz'])],
                              colors8[4:5], ['Local Position VZ'])
        data_plot.change_dataset('vehicle_gps_position')
        data_plot.add_graph([lambda data: ('vel_d_m_s', -data['vel_d_m_s'])], colors8[6:7], ['GPS VZ'])
    except:
        pass


    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)
    
    data_plot = DataPlot(data, plot_config, 'estimator_status', title='Output Tracking Errors',
                         changed_params=changed_params, x_range=x_range)
    data_plot.add_graph([lambda data: ('output_tracking_error[0]', 180.0/np.pi * data['output_tracking_error[0]']),
                         'output_tracking_error[1]', 'output_tracking_error[2]'], colors13[0:3],
                        ['Angle [deg]', 'Velocity [m/s]', 'Position [m]'])

    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)

    if data_plot.finalize() is not None: plots.append(data_plot)

    # Failsafe flags
    try:
        data_plot = DataPlot(data, plot_config, 'vehicle_status',
                             y_start=0, title='Failsafe Flags',
                             plot_height='normal', changed_params=changed_params,
                             x_range=x_range)
        data_plot.add_graph(['failsafe', 'failsafe_and_user_took_over'], [colors8[0], colors8[1]],
                            ['In Failsafe', 'User Took Over'])
        num_graphs = 2
        skip_if_always_set = ['auto_mission_missing', 'offboard_control_signal_lost']

        data_plot.change_dataset('failsafe_flags')
        if data_plot.dataset is not None:
            failsafe_flags = data_plot.dataset.data
            for failsafe_field in failsafe_flags:
                if failsafe_field == 'timestamp' or failsafe_field.startswith('mode_req_'):
                    continue
                cur_data = failsafe_flags[failsafe_field]
                # filter: show only the flags that are set at some point
                if np.amax(cur_data) >= 1:
                    if failsafe_field in skip_if_always_set and np.amin(cur_data) >= 1:
                        continue
                    data_plot.add_graph([failsafe_field], [colors8[num_graphs % 8]],
                                        [failsafe_field.replace('_', ' ')])
                    num_graphs += 1
            plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
            if data_plot.finalize() is not None: plots.append(data_plot)
    except (KeyError, IndexError) as error:
        print('Error in failsafe plot: '+str(error))


    # cpu load
    data_plot = DataPlot(data, plot_config, 'cpuload',
                         title='CPU & RAM', plot_height='small', y_range=Range1d(0, 1),
                         changed_params=changed_params, x_range=x_range)
    data_plot.add_graph(['ram_usage', 'load'], [colors3[1], colors3[2]],
                        ['RAM Usage', 'CPU Load'])
    data_plot.add_span('load', line_color=colors3[2])
    data_plot.add_span('ram_usage', line_color=colors3[1])
    plot_flight_modes_background(data_plot, flight_mode_changes, vtol_states)
    if data_plot.finalize() is not None: plots.append(data_plot)


    # sampling: time difference
    try:
        data_plot = DataPlot(data, plot_config, 'sensor_combined', y_range=Range1d(0, 25e3),
                             y_axis_label='[us]',
                             title='Sampling Regularity of Sensor Data', plot_height='small',
                             changed_params=changed_params, x_range=x_range)
        sensor_combined = ulog.get_dataset('sensor_combined').data
        sampling_diff = np.diff(sensor_combined['timestamp'])
        min_sampling_diff = np.amin(sampling_diff)

        plot_dropouts(data_plot.bokeh_plot, ulog.dropouts, min_sampling_diff)

        data_plot.add_graph([lambda data: ('timediff', np.append(sampling_diff, 0))],
                            [colors3[2]], ['delta t (between 2 logged samples)'])
        data_plot.change_dataset('estimator_status')
        data_plot.add_graph([lambda data: ('time_slip', data['time_slip']*1e6)],
                            [colors3[1]], ['Estimator time slip (cumulative)'])
        if data_plot.finalize() is not None: plots.append(data_plot)
    except:
        pass



    # exchange all DataPlot's with the bokeh_plot and handle parameter changes

    param_changes_button = Button(label="Hide Parameter Changes", width=170)
    param_change_labels = []
    # FIXME: this should be a CustomJS callback, not on the server. However this
    # did not work for me.
    def param_changes_button_clicked():
        """ callback to show/hide parameter changes """
        for label in param_change_labels:
            if label.visible:
                param_changes_button.label = 'Show Parameter Changes'
                label.visible = False
                label.text_alpha = 0 # label.visible does not work, so we use this instead
            else:
                param_changes_button.label = 'Hide Parameter Changes'
                label.visible = True
                label.text_alpha = 1
    param_changes_button.on_click(param_changes_button_clicked)


    jinja_plot_data = []
    for i in range(len(plots)):
        if plots[i] is None:
            plots[i] = column(param_changes_button, width=int(plot_width * 0.99))
        if isinstance(plots[i], DataPlot):
            if plots[i].param_change_label is not None:
                param_change_labels.append(plots[i].param_change_label)

            plot_title = plots[i].title
            plots[i] = plots[i].bokeh_plot

            fragment = 'Nav-'+plot_title.replace(' ', '-') \
                .replace('&', '_').replace('(', '').replace(')', '')
            jinja_plot_data.append({
                'model_id': plots[i].ref['id'],
                'fragment': fragment,
                'title': plot_title
                })


    # changed parameters
    plots.append(get_changed_parameters(ulog, plot_width))



    # information about which messages are contained in the log
# TODO: need to load all topics for this (-> log loading will take longer)
#       but if we load all topics and the log contains some (external) topics
#       with buggy timestamps, it will affect the plotting.
#    data_list_sorted = sorted(ulog.data_list, key=lambda d: d.name + str(d.multi_id))
#    table_text = []
#    for d in data_list_sorted:
#        message_size = sum([ULog.get_field_size(f.type_str) for f in d.field_data])
#        num_data_points = len(d.data['timestamp'])
#        table_text.append((d.name, str(d.multi_id), str(message_size), str(num_data_points),
#           str(message_size * num_data_points)))
#    topics_info = '<table><tr><th>Name</th><th>Topic instance</th><th>Message Size</th>' \
#            '<th>Number of data points</th><th>Total bytes</th></tr>' + ''.join(
#            ['<tr><td>'+'</td><td>'.join(list(x))+'</td></tr>' for x in table_text]) + '</table>'
#    topics_div = Div(text=topics_info, width=int(plot_width*0.9))
#    plots.append(column(topics_div, width=int(plot_width*0.9)))


    # log messages
    plots.append(get_logged_messages(ulog, plot_width))


    # console messages, perf & top output
    top_data = ''
    perf_data = ''
    console_messages = ''
    if 'boot_console_output' in ulog.msg_info_multiple_dict:
        console_output = ulog.msg_info_multiple_dict['boot_console_output'][0]
        console_output = escape(''.join(console_output))
        console_messages = '<p><pre>'+console_output+'</pre></p>'

    for state in ['pre', 'post']:
        if 'perf_top_'+state+'flight' in ulog.msg_info_multiple_dict:
            current_top_data = ulog.msg_info_multiple_dict['perf_top_'+state+'flight'][0]
            flight_data = escape('\n'.join(current_top_data))
            top_data += '<p>'+state.capitalize()+' Flight:<br/><pre>'+flight_data+'</pre></p>'
        if 'perf_counter_'+state+'flight' in ulog.msg_info_multiple_dict:
            current_perf_data = ulog.msg_info_multiple_dict['perf_counter_'+state+'flight'][0]
            flight_data = escape('\n'.join(current_perf_data))
            perf_data += '<p>'+state.capitalize()+' Flight:<br/><pre>'+flight_data+'</pre></p>'
    if 'perf_top_watchdog' in ulog.msg_info_multiple_dict:
        current_top_data = ulog.msg_info_multiple_dict['perf_top_watchdog'][0]
        flight_data = escape('\n'.join(current_top_data))
        top_data += '<p>Watchdog:<br/><pre>'+flight_data+'</pre></p>'

    additional_data_html = ''
    if len(console_messages) > 0:
        additional_data_html += '<h5>Console Output</h5>'+console_messages
    if len(top_data) > 0:
        additional_data_html += '<h5>Processes</h5>'+top_data
    if len(perf_data) > 0:
        additional_data_html += '<h5>Performance Counters</h5>'+perf_data
    if len(additional_data_html) > 0:
        # hide by default & use a button to expand
        additional_data_html = '''
<button id="show-additional-data-btn" class="btn btn-secondary" data-toggle="collapse" style="min-width:0;"
 data-target="#show-additional-data">Show additional Data</button>
<div id="show-additional-data" class="collapse">
{:}
</div>
'''.format(additional_data_html)
        curdoc().template_variables['additional_info'] = additional_data_html


    curdoc().template_variables['plots'] = jinja_plot_data

    return plots
