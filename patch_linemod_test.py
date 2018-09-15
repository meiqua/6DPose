import os
import sys
import time
import numpy as np
import cv2
import math
from pysixd import view_sampler, inout, misc
from pysixd.renderer import render
from params.dataset_params import get_dataset_params
from os.path import join
import copy
import linemodLevelup_pybind

from pysixd import renderer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

dataset = 'hinterstoisser'
# dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
# dataset = 'doumanoglou'
# dataset = 'toyotalight'

# mode = 'render_train'
mode = 'test'

dp = get_dataset_params(dataset)
detector = linemodLevelup_pybind.Detector(16, [4, 8], 16)  # min features; pyramid strides; num clusters

obj_ids = []  # for each obj
obj_ids_curr = range(1, dp['obj_count'] + 1)
if obj_ids:
    obj_ids_curr = set(obj_ids_curr).intersection(obj_ids)

scene_ids = []  # for each obj
im_ids = []  # obj's img
gt_ids = []  # multi obj in one img
scene_ids_curr = range(1, dp['scene_count'] + 1)
if scene_ids:
    scene_ids_curr = set(scene_ids_curr).intersection(scene_ids)

# mm
dep_range = 200  # max depth range of objects
dep_anchors = []  # depth to apply templates

dep_min = dp['test_obj_depth_range'][0]  # min depth of scene
dep_max = dp['test_obj_depth_range'][1]  # max depth of scene
dep_anchor_step = 1.2  # depth scale

# dep_min = 400  # min depth of scene
# dep_max = 1000  # max depth of scene
# dep_anchor_step = 1.2  # depth scale

current_dep = dep_min
while current_dep < dep_max:
    dep_anchors.append(int(current_dep))
    current_dep = current_dep*dep_anchor_step

print('\ndep anchors:\n {}, \ndep range: {}\n'.format(dep_anchors, dep_range))

top_level_path = os.path.dirname(os.path.abspath(__file__))
template_saved_to = join(dp['base_path'], 'linemod_render_up', '%s.yaml')
tempInfo_saved_to = join(dp['base_path'], 'linemod_render_up', '{:02d}_info_{}.yaml')
result_base_path = join(top_level_path, 'public', 'sixd_results', 'patch-linemod_'+dataset)

misc.ensure_dir(os.path.dirname(template_saved_to))
misc.ensure_dir(os.path.dirname(tempInfo_saved_to))
misc.ensure_dir(result_base_path)

if mode == 'render_train':
    start_time = time.time()

    im_size = dp['cam']['im_size']
    shape = (im_size[1], im_size[0])

    # Frame buffer object, bind here to avoid memory leak, maybe?
    window = renderer.app.Window(visible=False)
    color_buf = np.zeros((shape[0], shape[1], 4), np.float32).view(renderer.gloo.TextureFloat2D)
    depth_buf = np.zeros((shape[0], shape[1]), np.float32).view(renderer.gloo.DepthTexture)
    fbo = renderer.gloo.FrameBuffer(color=color_buf, depth=depth_buf)
    fbo.activate()

    for obj_id in obj_ids_curr:
        azimuth_range = dp['test_obj_azimuth_range']
        elev_range = dp['test_obj_elev_range']
        min_n_views = 233
        clip_near = 10  # [mm]
        clip_far = 10000  # [mm]
        ambient_weight = 0.8  # Weight of ambient light [0, 1]
        shading = 'phong'  # 'flat', 'phong'

        # Load model
        model_path = dp['model_mpath'].format(obj_id)
        model = inout.load_ply(model_path)

        # Load model texture
        if dp['model_texture_mpath']:
            model_texture_path = dp['model_texture_mpath'].format(obj_id)
            model_texture = inout.load_im(model_texture_path)
        else:
            model_texture = None

        ######################################################
        # prepare renderer rather than rebuilding every time

        texture = model_texture
        surf_color = None
        mode = 'rgb+depth'
        K = dp['cam']['K']

        assert ({'pts', 'faces'}.issubset(set(model.keys())))
        # Set texture / color of vertices
        if texture is not None:
            if texture.max() > 1.0:
                texture = texture.astype(np.float32) / 255.0
            texture = np.flipud(texture)
            texture_uv = model['texture_uv']
            colors = np.zeros((model['pts'].shape[0], 3), np.float32)
        else:
            texture_uv = np.zeros((model['pts'].shape[0], 2), np.float32)
            if not surf_color:
                if 'colors' in model.keys():
                    assert (model['pts'].shape[0] == model['colors'].shape[0])
                    colors = model['colors']
                    if colors.max() > 1.0:
                        colors /= 255.0  # Color values are expected in range [0, 1]
                else:
                    colors = np.ones((model['pts'].shape[0], 3), np.float32) * 0.5
            else:
                colors = np.tile(list(surf_color) + [1.0], [model['pts'].shape[0], 1])

        # Set the vertex data
        if mode == 'depth':
            vertices_type = [('a_position', np.float32, 3),
                             ('a_color', np.float32, colors.shape[1])]
            vertices = np.array(list(zip(model['pts'], colors)), vertices_type)
        else:
            if shading == 'flat':
                vertices_type = [('a_position', np.float32, 3),
                                 ('a_color', np.float32, colors.shape[1]),
                                 ('a_texcoord', np.float32, 2)]
                vertices = np.array(list(zip(model['pts'], colors, texture_uv)),
                                    vertices_type)
            else:  # shading == 'phong'
                vertices_type = [('a_position', np.float32, 3),
                                 ('a_normal', np.float32, 3),
                                 ('a_color', np.float32, colors.shape[1]),
                                 ('a_texcoord', np.float32, 2)]
                vertices = np.array(list(zip(model['pts'], model['normals'],
                                             colors, texture_uv)), vertices_type)

        # Projection matrix
        mat_proj = renderer._compute_calib_proj(K, 0, 0, im_size[0], im_size[1], clip_near, clip_far)

        # Model matrix
        mat_model = np.eye(4, dtype=np.float32)  # From object space to world space

        # Create buffers
        vertex_buffer = vertices.view(renderer.gloo.VertexBuffer)
        index_buffer = model['faces'].flatten().astype(np.uint32).view(renderer.gloo.IndexBuffer)

        bg_color = (0.0, 0.0, 0.0, 0.0)
        program_dep = renderer.gloo.Program(renderer._depth_vertex_code, renderer._depth_fragment_code)
        program_dep.bind(vertex_buffer)

        # Set shader for the selected shading
        if shading == 'flat':
            color_fragment_code = renderer._color_fragment_flat_code
        else:  # 'phong'
            color_fragment_code = renderer._color_fragment_phong_code

        program = renderer.gloo.Program(renderer._color_vertex_code, color_fragment_code)
        program.bind(vertex_buffer)
        program['u_light_eye_pos'] = [0, 0, 0]  # Camera origin
        program['u_light_ambient_w'] = ambient_weight
        if texture is not None:
            program['u_use_texture'] = int(True)
            program['u_texture'] = texture
        else:
            program['u_use_texture'] = int(False)
            program['u_texture'] = np.zeros((1, 1, 4), np.float32)

        # OpenGL setup
        renderer.gl.glEnable(renderer.gl.GL_DEPTH_TEST)
        renderer.gl.glViewport(0, 0, shape[1], shape[0])
        renderer.gl.glDisable(renderer.gl.GL_CULL_FACE)
        ######################################################

        # in our test, for complex objects fast-train performs badly...
        fast_train = False  # just scale templates
        if fast_train:
            # Sample views

            # with camera tilt
            views, views_level = view_sampler.sample_views(min_n_views, dep_anchors[0],
                                                           azimuth_range, elev_range,
                                                           tilt_range=(-math.pi, math.pi),
                                                           tilt_step=math.pi / 8)

            print('Sampled views: ' + str(len(views)))

            templateInfo_radius = dict()
            for dep in dep_anchors:
                templateInfo_radius[dep] = dict()

            # Render the object model from all the views
            for view_id, view in enumerate(views):

                if view_id % 10 == 0:
                    print('obj,radius,view: ' + str(obj_id) +
                          ',' + str(dep_anchors[0]) + ',' + str(view_id) + ', view_id: ', view_id)

                # Render depth image
                # depth = render(model, dp['cam']['im_size'], dp['cam']['K'],
                #                view['R'], view['t'],
                #                clip_near, clip_far, mode='depth')

                mat_view = np.eye(4, dtype=np.float32)  # From world space to eye space
                mat_view[:3, :3] = view['R']
                mat_view[:3, 3] = view['t'].squeeze()
                yz_flip = np.eye(4, dtype=np.float32)
                yz_flip[1, 1], yz_flip[2, 2] = -1, -1
                mat_view = yz_flip.dot(mat_view)  # OpenCV to OpenGL camera system
                mat_view = mat_view.T  # OpenGL expects column-wise matrix format
                depth = renderer.draw_depth(shape, vertex_buffer, index_buffer, mat_model,
                                            mat_view, mat_proj)

                # Convert depth so it is in the same units as the real test images
                depth /= dp['cam']['depth_scale']
                depth = depth.astype(np.uint16)

                # Render RGB image
                # rgb = render(model, dp['cam']['im_size'], dp['cam']['K'], view['R'], view['t'],
                #              clip_near, clip_far, texture=model_texture,
                #              ambient_weight=ambient_weight, shading=shading,
                #              mode='rgb')

                rgb = renderer.draw_color(shape, vertex_buffer, index_buffer, texture, mat_model,
                                          mat_view, mat_proj, ambient_weight, bg_color, shading)

                rgb = cv2.resize(rgb, dp['cam']['im_size'], interpolation=cv2.INTER_AREA)

                # have read rgb, depth, pose, obj_bb, obj_id here

                rows = np.any(depth, axis=1)
                cols = np.any(depth, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]

                mask = (depth > 0).astype(np.uint8) * 255

                visual = False
                if visual:
                    cv2.namedWindow('rgb')
                    cv2.imshow('rgb', rgb)
                    cv2.waitKey(1000)

                success = detector.addTemplate([rgb, depth], '{:02d}_template_{}'.format(obj_id, dep_anchors[0]),
                                               mask, dep_anchors)
                del rgb, depth, mask

                print('success: {}'.format(success))
                for i in range(len(dep_anchors)):
                    if success[i] != -1:
                        aTemplateInfo = dict()
                        aTemplateInfo['cam_K'] = copy.deepcopy(dp['cam']['K'])
                        aTemplateInfo['cam_R_w2c'] = copy.deepcopy(view['R'])
                        aTemplateInfo['cam_t_w2c'] = copy.deepcopy(view['t'])
                        aTemplateInfo['cam_t_w2c'][2] = dep_anchors[i]

                        templateInfo = templateInfo_radius[dep_anchors[i]]
                        templateInfo[success[i]] = aTemplateInfo

            for radius in dep_anchors:
                inout.save_info(tempInfo_saved_to.format(obj_id, radius), templateInfo_radius[radius])

            detector.writeClasses(template_saved_to)
            #  clear to save RAM
            detector.clear_classes()
        else:
            for radius in dep_anchors:
                # Sample views

                # with camera tilt
                # tilt_factor = (80 / 180)
                tilt_factor = 1
                views, views_level = view_sampler.sample_views(min_n_views, radius,
                                                               azimuth_range, elev_range,
                                                               tilt_range=(-math.pi * tilt_factor,
                                                                           math.pi * tilt_factor),
                                                               tilt_step=math.pi / 8)
                print('Sampled views: ' + str(len(views)))

                templateInfo = dict()

                # Render the object model from all the views
                for view_id, view in enumerate(views):
                    window.clear()

                    if view_id % 50 == 0:
                        print(dataset + ' obj,radius,view: ' + str(obj_id) +
                              ',' + str(radius) + ',' + str(view_id) + ', view_id: ', view_id)
                        # cv2.waitKey(0)

                    # Render depth image
                    # depth = render(model, dp['cam']['im_size'], dp['cam']['K'],
                    #                view['R'], view['t'],
                    #                clip_near, clip_far, mode='depth')

                    ################################################################
                    mat_view = np.eye(4, dtype=np.float32)  # From world space to eye space
                    mat_view[:3, :3] = view['R']
                    mat_view[:3, 3] = view['t'].squeeze()
                    yz_flip = np.eye(4, dtype=np.float32)
                    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
                    mat_view = yz_flip.dot(mat_view)  # OpenCV to OpenGL camera system
                    mat_view = mat_view.T  # OpenGL expects column-wise matrix format

                    renderer.gl.glClearColor(0.0, 0.0, 0.0, 0.0)
                    renderer.gl.glClear(renderer.gl.GL_COLOR_BUFFER_BIT | renderer.gl.GL_DEPTH_BUFFER_BIT)

                    program_dep['u_mv'] = renderer._compute_model_view(mat_model, mat_view)
                    program_dep['u_mvp'] = renderer._compute_model_view_proj(mat_model, mat_view, mat_proj)
                    program_dep.draw(renderer.gl.GL_TRIANGLES, index_buffer)

                    # Retrieve the contents of the FBO texture
                    depth = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
                    renderer.gl.glReadPixels(0, 0, shape[1], shape[0], renderer.gl.GL_RGBA, renderer.gl.GL_FLOAT, depth)
                    depth.shape = shape[0], shape[1], 4
                    depth = depth[::-1, :]
                    depth = depth[:, :, 0]  # Depth is saved in the first channel
                    #################################################################

                    # Convert depth so it is in the same units as the real test images
                    depth *= dp['cam']['depth_scale']
                    depth = depth.astype(np.uint16)

                    # # Render RGB image
                    ##################################################################
                    renderer.gl.glClearColor(bg_color[0], bg_color[1], bg_color[2], bg_color[3])
                    renderer.gl.glClear(renderer.gl.GL_COLOR_BUFFER_BIT | renderer.gl.GL_DEPTH_BUFFER_BIT)

                    program['u_mv'] = renderer._compute_model_view(mat_model, mat_view)
                    program['u_nm'] = renderer._compute_normal_matrix(mat_model, mat_view)
                    program['u_mvp'] = renderer._compute_model_view_proj(mat_model, mat_view, mat_proj)
                    program.draw(renderer.gl.GL_TRIANGLES, index_buffer)

                    # Retrieve the contents of the FBO texture
                    rgb = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
                    renderer.gl.glReadPixels(0, 0, shape[1], shape[0], renderer.gl.GL_RGBA, renderer.gl.GL_FLOAT, rgb)
                    rgb.shape = shape[0], shape[1], 4
                    rgb = rgb[::-1, :]
                    rgb = np.round(rgb[:, :, :3] * 255).astype(np.uint8)  # Convert to [0, 255]
                    ##################################################################

                    # rgb = renderer.draw_color(shape, vertex_buffer, index_buffer, texture, mat_model,
                    #                          mat_view, mat_proj, ambient_weight, bg_color, shading)

                    rgb = cv2.resize(rgb, dp['cam']['im_size'], interpolation=cv2.INTER_AREA)

                    K = dp['cam']['K']
                    R = view['R']
                    t = view['t']
                    # have read rgb, depth, pose, obj_bb, obj_id here

                    rows = np.any(depth, axis=1)
                    cols = np.any(depth, axis=0)
                    ymin, ymax = np.where(rows)[0][[0, -1]]
                    xmin, xmax = np.where(cols)[0][[0, -1]]

                    aTemplateInfo = dict()
                    aTemplateInfo['cam_K'] = K
                    aTemplateInfo['cam_R_w2c'] = R
                    aTemplateInfo['cam_t_w2c'] = t
                    aTemplateInfo['width'] = int(xmax - xmin)
                    aTemplateInfo['height'] = int(ymax - ymin)

                    mask = (depth > 0).astype(np.uint8) * 255

                    visual = False
                    if visual:
                        cv2.imshow('rgb', rgb)
                        cv2.imshow('mask', mask)
                        cv2.waitKey(0)

                    success = detector.addTemplate([rgb, depth], '{:02d}_template_{}'.format(obj_id, radius), mask, [])
                    print('success {}'.format(success[0]))

                    del rgb, depth, mask

                    if success[0] != -1:
                        templateInfo[success[0]] = aTemplateInfo

                inout.save_info(tempInfo_saved_to.format(obj_id, radius), templateInfo)
                detector.writeClasses(template_saved_to)
                #  clear to save RAM
                detector.clear_classes()

    fbo.deactivate()
    window.close()

    elapsed_time = time.time() - start_time
    print('train time: {}\n'.format(elapsed_time))

if mode == 'test':
    poseRefine = linemodLevelup_pybind.poseRefine()

    im_size = dp['test_im_size']
    shape = (im_size[1], im_size[0])
    print('test img size: {}'.format(shape))

    # Frame buffer object, bind here to avoid memory leak, maybe?
    window = renderer.app.Window(visible=False)
    color_buf = np.zeros((shape[0], shape[1], 4), np.float32).view(renderer.gloo.TextureFloat2D)
    depth_buf = np.zeros((shape[0], shape[1]), np.float32).view(renderer.gloo.DepthTexture)
    fbo = renderer.gloo.FrameBuffer(color=color_buf, depth=depth_buf)
    fbo.activate()

    use_image_subset = True
    if use_image_subset:
        im_ids_sets = inout.load_yaml(dp['test_set_fpath'])
    else:
        im_ids_sets = None

    for scene_id in scene_ids_curr:
        # obj_id_in_scene = 5  # for different obj in same scene
        obj_id_in_scene = scene_id
        # Load scene info and gt poses
        print('#'*20)
        print('\nreading detector template & info, obj: {}'.format(obj_id_in_scene))
        misc.ensure_dir(join(result_base_path, '{:02d}'.format(scene_id)))
        scene_info = inout.load_info(dp['scene_info_mpath'].format(scene_id))
        scene_gt = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))
        model = inout.load_ply(dp['model_mpath'].format(obj_id_in_scene))

        ######################################################
        # prepare renderer rather than rebuilding every time

        clip_near = 10  # [mm]
        clip_far = 10000  # [mm]
        ambient_weight = 0.8

        surf_color = None
        mode = 'rgb+depth'
        K = dp['cam']['K']
        shading = 'phong'

        # Load model texture
        if dp['model_texture_mpath']:
            model_texture_path = dp['model_texture_mpath'].format(scene_id)
            model_texture = inout.load_im(model_texture_path)
        else:
            model_texture = None

        texture = model_texture

        assert ({'pts', 'faces'}.issubset(set(model.keys())))
        # Set texture / color of vertices
        if texture is not None:
            if texture.max() > 1.0:
                texture = texture.astype(np.float32) / 255.0
            texture = np.flipud(texture)
            texture_uv = model['texture_uv']
            colors = np.zeros((model['pts'].shape[0], 3), np.float32)
        else:
            texture_uv = np.zeros((model['pts'].shape[0], 2), np.float32)
            if not surf_color:
                if 'colors' in model.keys():
                    assert (model['pts'].shape[0] == model['colors'].shape[0])
                    colors = model['colors']
                    if colors.max() > 1.0:
                        colors /= 255.0  # Color values are expected in range [0, 1]
                else:
                    colors = np.ones((model['pts'].shape[0], 3), np.float32) * 0.5
            else:
                colors = np.tile(list(surf_color) + [1.0], [model['pts'].shape[0], 1])

        # Set the vertex data
        if mode == 'depth':
            vertices_type = [('a_position', np.float32, 3),
                             ('a_color', np.float32, colors.shape[1])]
            vertices = np.array(list(zip(model['pts'], colors)), vertices_type)
        else:
            if shading == 'flat':
                vertices_type = [('a_position', np.float32, 3),
                                 ('a_color', np.float32, colors.shape[1]),
                                 ('a_texcoord', np.float32, 2)]
                vertices = np.array(list(zip(model['pts'], colors, texture_uv)),
                                    vertices_type)
            else:  # shading == 'phong'
                vertices_type = [('a_position', np.float32, 3),
                                 ('a_normal', np.float32, 3),
                                 ('a_color', np.float32, colors.shape[1]),
                                 ('a_texcoord', np.float32, 2)]
                vertices = np.array(list(zip(model['pts'], model['normals'],
                                             colors, texture_uv)), vertices_type)

        # Model matrix
        mat_model = np.eye(4, dtype=np.float32)  # From object space to world space

        # Create buffers
        vertex_buffer = vertices.view(renderer.gloo.VertexBuffer)
        index_buffer = model['faces'].flatten().astype(np.uint32).view(renderer.gloo.IndexBuffer)

        bg_color = (0.0, 0.0, 0.0, 0.0)
        program_dep = renderer.gloo.Program(renderer._depth_vertex_code, renderer._depth_fragment_code)
        program_dep.bind(vertex_buffer)

        # Set shader for the selected shading
        if shading == 'flat':
            color_fragment_code = renderer._color_fragment_flat_code
        else:  # 'phong'
            color_fragment_code = renderer._color_fragment_phong_code

        program = renderer.gloo.Program(renderer._color_vertex_code, color_fragment_code)
        program.bind(vertex_buffer)
        program['u_light_eye_pos'] = [0, 0, 0]  # Camera origin
        program['u_light_ambient_w'] = ambient_weight
        if texture is not None:
            program['u_use_texture'] = int(True)
            program['u_texture'] = texture
        else:
            program['u_use_texture'] = int(False)
            program['u_texture'] = np.zeros((1, 1, 4), np.float32)

        # OpenGL setup
        renderer.gl.glEnable(renderer.gl.GL_DEPTH_TEST)
        renderer.gl.glViewport(0, 0, shape[1], shape[0])
        renderer.gl.glDisable(renderer.gl.GL_CULL_FACE)
        ######################################################

        template_read_classes = []
        detector.clear_classes()
        for radius in dep_anchors:
            template_read_classes.append('{:02d}_template_{}'.format(obj_id_in_scene, radius))
        detector.readClasses(template_read_classes, template_saved_to)

        print('num templs: {}'.format(detector.numTemplates()))

        templateInfo = dict()
        for radius in dep_anchors:
            key = tempInfo_saved_to.format(obj_id_in_scene, radius)
            aTemplateInfo = inout.load_info(key)
            key = os.path.basename(key)
            key = os.path.splitext(key)[0]
            key = key.replace('info', 'template')
            templateInfo[key] = aTemplateInfo

        # Considered subset of images for the current scene
        if im_ids_sets is not None:
            im_ids_curr = im_ids_sets[scene_id]
        else:
            im_ids_curr = sorted(scene_info.keys())

        if im_ids:
            im_ids_curr = set(im_ids_curr).intersection(im_ids)

        # active ratio should be higher for simple objects
        # we adjust this factor according to candidates size
        trick_factor = 1
        factor_lock = 0
        base_active_ratio = 0.6
        for im_id in im_ids_curr:

            if factor_lock < 20:  # avoid overflow(may never happen)
                factor_lock += 1

            start_time = time.time()

            print('#'*20)
            print('\nscene: {}, im: {}'.format(scene_id, im_id))

            K = scene_info[im_id]['cam_K']
            # Load the images
            rgb = inout.load_im(dp['test_rgb_mpath'].format(scene_id, im_id))
            depth = inout.load_depth(dp['test_depth_mpath'].format(scene_id, im_id))
            depth *= dp['cam']['depth_scale']
            depth = depth.astype(np.uint16)  # [mm]
            im_size = (depth.shape[1], depth.shape[0])

            match_ids = list()

            for radius in dep_anchors:
                match_ids.append('{:02d}_template_{}'.format(obj_id_in_scene, radius))

            # srcs, score for one part, active ratio
            matches = detector.match([rgb, depth], 60, base_active_ratio*trick_factor,
                                     match_ids, dep_anchors, dep_range, masks=[])

            if len(matches) > 0:
                aTemplateInfo = templateInfo[matches[0].class_id]
                render_K = aTemplateInfo[0]['cam_K']

            print('candidates size before refine & nms: {}\n'.format(len(matches)))

            dets = []
            Rs = []
            Ts = []
            icp_scores = []
            local_refine_start = time.time()
            icp_time = 0

            top50_local_refine = 150  # avoid too many for simple obj,
            # we observed more than 1000 when active ratio too low
            if top50_local_refine >= len(matches):
                top50_local_refine = len(matches)
            else:
                if factor_lock <= 10:
                    trick_factor = trick_factor + 0.03
                    if trick_factor > 1 / base_active_ratio:
                        trick_factor = 1 / base_active_ratio
                    if trick_factor < base_active_ratio / 2:
                        trick_factor = base_active_ratio / 2
                    print('active ratio too low, increase trick factor: {}'.format(trick_factor))

            for i in range(top50_local_refine):
                match = matches[i]

                aTemplateInfo = templateInfo[match.class_id]
                K_match = aTemplateInfo[match.template_id]['cam_K']
                R_match = aTemplateInfo[match.template_id]['cam_R_w2c']
                t_match = aTemplateInfo[match.template_id]['cam_t_w2c']

                ################################################################
                R_in = R_match
                t_in = t_match
                K_in = K_match

                mat_view = np.eye(4, dtype=np.float32)  # From world space to eye space
                mat_view[:3, :3] = R_in
                mat_view[:3, 3] = t_in.squeeze()
                yz_flip = np.eye(4, dtype=np.float32)
                yz_flip[1, 1], yz_flip[2, 2] = -1, -1
                mat_view = yz_flip.dot(mat_view)  # OpenCV to OpenGL camera system
                mat_view = mat_view.T  # OpenGL expects column-wise matrix format

                window.clear()
                renderer.gl.glClearColor(0.0, 0.0, 0.0, 0.0)
                renderer.gl.glClear(renderer.gl.GL_COLOR_BUFFER_BIT | renderer.gl.GL_DEPTH_BUFFER_BIT)

                # Projection matrix
                mat_proj = renderer._compute_calib_proj(K_in, 0, 0, im_size[0], im_size[1], clip_near, clip_far)

                program_dep['u_mv'] = renderer._compute_model_view(mat_model, mat_view)
                program_dep['u_mvp'] = renderer._compute_model_view_proj(mat_model, mat_view, mat_proj)
                program_dep.draw(renderer.gl.GL_TRIANGLES, index_buffer)

                # Retrieve the contents of the FBO texture
                depth_out = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
                renderer.gl.glReadPixels(0, 0, shape[1], shape[0], renderer.gl.GL_RGBA, renderer.gl.GL_FLOAT, depth_out)
                depth_out.shape = shape[0], shape[1], 4
                depth_out = depth_out[::-1, :]
                depth_out = depth_out[:, :, 0]  # Depth is saved in the first channel

                window.clear()
                renderer.gl.glClearColor(bg_color[0], bg_color[1], bg_color[2], bg_color[3])
                renderer.gl.glClear(renderer.gl.GL_COLOR_BUFFER_BIT | renderer.gl.GL_DEPTH_BUFFER_BIT)

                program['u_mv'] = renderer._compute_model_view(mat_model, mat_view)
                program['u_nm'] = renderer._compute_normal_matrix(mat_model, mat_view)
                program['u_mvp'] = renderer._compute_model_view_proj(mat_model, mat_view, mat_proj)
                program.draw(renderer.gl.GL_TRIANGLES, index_buffer)

                # Retrieve the contents of the FBO texture
                rgb_out = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
                renderer.gl.glReadPixels(0, 0, shape[1], shape[0], renderer.gl.GL_RGBA, renderer.gl.GL_FLOAT, rgb_out)
                rgb_out.shape = shape[0], shape[1], 4
                rgb_out = rgb_out[::-1, :]
                rgb_out = np.round(rgb_out[:, :, :3] * 255).astype(np.uint8)  # Convert to [0, 255]
                #################################################################

                depth_ren = depth_out

                icp_start = time.time()
                # make sure data type is consistent
                poseRefine.process(depth.astype(np.uint16), depth_ren.astype(np.uint16), K.astype(np.float32),
                                   K_match.astype(np.float32), R_match.astype(np.float32), t_match.astype(np.float32)
                                   , match.x, match.y)
                icp_time += (time.time() - icp_start)

                refinedR = poseRefine.result_refined[0:3, 0:3]
                refinedT = poseRefine.result_refined[0:3, 3]
                refinedT = np.reshape(refinedT, (3,))*1000
                score = 1/(poseRefine.inlier_rmse + 0.01)

                if poseRefine.fitness < base_active_ratio*trick_factor or poseRefine.inlier_rmse > 0.01:
                    continue

                # simple color check
                mask_model = depth_out > 0
                mask_model = mask_model.astype(np.uint8)
                mask_model = cv2.erode(mask_model, np.ones((5,5),np.uint8))
                rgb_mask = np.dstack([mask_model] * 3)
                rgb_model = rgb_out*rgb_mask
                hsv_model = cv2.cvtColor(rgb_model, cv2.COLOR_BGR2HSV)
                avg_v_model = np.sum(hsv_model[:, :, 2]*mask_model)/np.sum(mask_model)

                mask_scene = np.zeros_like(mask_model)
                #bbox, note, variable name may not be right
                rows = np.any(mask_model, axis=1)
                cols = np.any(mask_model, axis=0)
                rmin, rmax = np.where(cols)[0][[0, -1]]
                cmin, cmax = np.where(rows)[0][[0, -1]]

                mask_scene[match.y:(match.y+cmax-cmin), match.x:(match.x+rmax-rmin)] = mask_model[cmin:cmax, rmin:rmax]
                rgb_mask_scene = np.dstack([mask_scene] * 3)
                rgb_scene = rgb*rgb_mask_scene
                hsv_scene = cv2.cvtColor(rgb_scene, cv2.COLOR_BGR2HSV)
                avg_v_scene = np.sum(hsv_scene[:, :, 2]*mask_scene)/np.sum(mask_scene)

                # print('v1, v2: {}, {}'.format(avg_v_model, avg_v_scene))
                # cv2.imshow('rgb', rgb_scene)
                # cv2.waitKey(0)

                if avg_v_scene/avg_v_model < 0.6 or avg_v_scene/avg_v_model > 1/0.6:
                    continue

                Rs.append(refinedR)
                Ts.append(refinedT)
                icp_scores.append(score)

                templ = detector.getTemplates(match.class_id, match.template_id)
                dets.append([match.x, match.y, match.x + templ[0].width, match.y + templ[0].height, score])

            idx = []
            if len(dets) > 0:
                idx = nms(np.array(dets), 0.4)

            print('candidates size after refine & nms: {}\n'.format(len(idx)))

            top5 = 10

            if len(idx) > int(top5*1.5):  # we don't want too many
                if factor_lock <= 10:
                    trick_factor = trick_factor + 0.03
                    if trick_factor > 1 / base_active_ratio:
                        trick_factor = 1 / base_active_ratio
                    if trick_factor < base_active_ratio / 2:
                        trick_factor = base_active_ratio / 2
                    print('active ratio too low, increase trick factor: {}'.format(trick_factor))

            if top5 > len(idx):
                top5 = len(idx)

            if top5 == 0:
                if factor_lock <= 10:
                    trick_factor = trick_factor - 0.03
                    if trick_factor > 1 / base_active_ratio:
                        trick_factor = 1 / base_active_ratio
                    if trick_factor < base_active_ratio / 2:
                        trick_factor = base_active_ratio / 2
                    print('active ratio too high, decrease trick factor: {}'.format(trick_factor))

            result = {}
            result_ests = []
            result_name = join(result_base_path, '{:02d}'.format(scene_id), '{:04d}_{:02d}.yml'.format(im_id, scene_id))

            for i in range(top5):
                e = dict()
                e['R'] = Rs[idx[i]]
                e['t'] = Ts[idx[i]]
                e['score'] = icp_scores[idx[i]]  # mse is smaller better, so 1/
                result_ests.append(e)

            print('local refine time: {}s'.format(time.time() - local_refine_start))
            print('icp time: {}s'.format(icp_time))

            matching_time = time.time() - start_time
            print('matching time: {}s'.format(matching_time))

            result['ests'] = result_ests
            inout.save_results_sixd17(result_name, result, matching_time)

            scores = []
            for e in result_ests:
                scores.append(e['score'])
            sort_index = np.argsort(np.array(scores))  # ascending

            # draw results
            render_rgb = rgb
            for i in range(len(scores)):
                render_R = result_ests[sort_index[i]]['R']
                render_t = result_ests[sort_index[i]]['t']

                ################################################################
                R_in = render_R
                t_in = render_t
                K_in = K

                mat_view = np.eye(4, dtype=np.float32)  # From world space to eye space
                mat_view[:3, :3] = R_in
                mat_view[:3, 3] = t_in.squeeze()
                yz_flip = np.eye(4, dtype=np.float32)
                yz_flip[1, 1], yz_flip[2, 2] = -1, -1
                mat_view = yz_flip.dot(mat_view)  # OpenCV to OpenGL camera system
                mat_view = mat_view.T  # OpenGL expects column-wise matrix format

                window.clear()
                renderer.gl.glClearColor(0.0, 0.0, 0.0, 0.0)
                renderer.gl.glClear(renderer.gl.GL_COLOR_BUFFER_BIT | renderer.gl.GL_DEPTH_BUFFER_BIT)

                # Projection matrix
                mat_proj = renderer._compute_calib_proj(K_in, 0, 0, im_size[0], im_size[1], clip_near, clip_far)

                program_dep['u_mv'] = renderer._compute_model_view(mat_model, mat_view)
                program_dep['u_mvp'] = renderer._compute_model_view_proj(mat_model, mat_view, mat_proj)
                program['u_nm'] = renderer._compute_normal_matrix(mat_model, mat_view)
                program_dep.draw(renderer.gl.GL_TRIANGLES, index_buffer)

                # Retrieve the contents of the FBO texture
                depth_out = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
                renderer.gl.glReadPixels(0, 0, shape[1], shape[0], renderer.gl.GL_RGBA, renderer.gl.GL_FLOAT, depth_out)
                depth_out.shape = shape[0], shape[1], 4
                depth_out = depth_out[::-1, :]
                depth_out = depth_out[:, :, 0]  # Depth is saved in the first channel

                window.clear()
                renderer.gl.glClearColor(bg_color[0], bg_color[1], bg_color[2], bg_color[3])
                renderer.gl.glClear(renderer.gl.GL_COLOR_BUFFER_BIT | renderer.gl.GL_DEPTH_BUFFER_BIT)

                program['u_mv'] = renderer._compute_model_view(mat_model, mat_view)
                program['u_nm'] = renderer._compute_normal_matrix(mat_model, mat_view)
                program['u_mvp'] = renderer._compute_model_view_proj(mat_model, mat_view, mat_proj)
                program.draw(renderer.gl.GL_TRIANGLES, index_buffer)

                # Retrieve the contents of the FBO texture
                rgb_out = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
                renderer.gl.glReadPixels(0, 0, shape[1], shape[0], renderer.gl.GL_RGBA, renderer.gl.GL_FLOAT, rgb_out)
                rgb_out.shape = shape[0], shape[1], 4
                rgb_out = rgb_out[::-1, :]
                rgb_out = np.round(rgb_out[:, :, :3] * 255).astype(np.uint8)  # Convert to [0, 255]
                #################################################################

                render_depth = depth_out
                render_rgb_new = rgb_out

                visible_mask = render_depth < depth
                mask = render_depth > 0
                mask = mask.astype(np.uint8)
                rgb_mask = np.dstack([mask] * 3)
                render_rgb = render_rgb * (1 - rgb_mask) + render_rgb_new * rgb_mask

                draw_axis(render_rgb, render_R, render_t, K)

                if i == len(scores) - 1:  # best result
                    draw_axis(rgb, render_R, render_t, K)

            visual = True
            # visual = False
            if visual:
                cv2.imshow('rgb_top1', rgb)
                cv2.imshow('rgb_render', render_rgb)
                cv2.waitKey(100)

    fbo.deactivate()
    window.close()

print('end line')