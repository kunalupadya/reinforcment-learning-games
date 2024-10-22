from gym_snake.register import register

# for num_players in ['', '2s', '3s', '4s']:
#     for style in ['DeadApple', '', 'Expand', '4a']:
#         for grid_size in ['4x4', '8x8', '16x16']:
#             for grid_type in ['', 'Hex']:
#                 env_id = '-'.join(['Snake', grid_type, grid_size, style, num_players]) + '-v0'.replace('--', '-')
#                 entry_point = 'gym_snake.envs:' + '_'.join(['Snake', grid_type, grid_size, style, num_players]).replace('--', '-')
#                 print("register(")
#                 print("    id='" + env_id + "',")
#                 print("    entry_point='" + entry_point + "'")
#                 print(")")
#
#                 pass

register(
    id='Snake-4x4-DeadApple-v0',
    entry_point='gym_snake.envs:Snake_4x4_DeadApple'
)
register(
    id='Snake-Hex-4x4-DeadApple-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4_DeadApple'
)
register(
    id='Snake-8x8-DeadApple-v0',
    entry_point='gym_snake.envs:Snake_8x8_DeadApple'
)
register(
    id='Snake-Hex-8x8-DeadApple-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8_DeadApple'
)
register(
    id='Snake-16x16-DeadApple-v0',
    entry_point='gym_snake.envs:Snake_16x16_DeadApple'
)
register(
    id='Snake-Hex-16x16-DeadApple-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16_DeadApple'
)
register(
    id='Snake-4x4-v0',
    entry_point='gym_snake.envs:Snake_4x4'
)
register(
    id='Snake-Hex-4x4-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4'
)
register(
    id='Snake-8x8-v0',
    entry_point='gym_snake.envs:Snake_8x8'
)
register(
    id='Snake-Hex-8x8-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8'
)
register(
    id='Snake-16x16-v0',
    entry_point='gym_snake.envs:Snake_16x16'
)
register(
    id='Snake-42x42-v0',
    entry_point='gym_snake.envs:Snake_42x42'
)
register(
    id='Snake-Hex-16x16-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16'
)
register(
    id='Snake-4x4-Expand-v0',
    entry_point='gym_snake.envs:Snake_4x4_Expand'
)
register(
    id='Snake-Hex-4x4-Expand-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4_Expand'
)
register(
    id='Snake-8x8-Expand-v0',
    entry_point='gym_snake.envs:Snake_8x8_Expand'
)
register(
    id='Snake-Hex-8x8-Expand-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8_Expand'
)
register(
    id='Snake-16x16-Expand-v0',
    entry_point='gym_snake.envs:Snake_16x16_Expand'
)
register(
    id='Snake-Hex-16x16-Expand-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16_Expand'
)
register(
    id='Snake-4x4-4a-v0',
    entry_point='gym_snake.envs:Snake_4x4_4a'
)
register(
    id='Snake-Hex-4x4-4a-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4_4a'
)
register(
    id='Snake-8x8-4a-v0',
    entry_point='gym_snake.envs:Snake_8x8_4a'
)
register(
    id='Snake-Hex-8x8-4a-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8_4a'
)
register(
    id='Snake-16x16-4a-v0',
    entry_point='gym_snake.envs:Snake_16x16_4a'
)
register(
    id='Snake-Hex-16x16-4a-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16_4a'
)
register(
    id='Snake-4x4-DeadApple-2s-v0',
    entry_point='gym_snake.envs:Snake_4x4_DeadApple_2s'
)
register(
    id='Snake-Hex-4x4-DeadApple-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4_DeadApple_2s'
)
register(
    id='Snake-8x8-DeadApple-2s-v0',
    entry_point='gym_snake.envs:Snake_8x8_DeadApple_2s'
)
register(
    id='Snake-Hex-8x8-DeadApple-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8_DeadApple_2s'
)
register(
    id='Snake-16x16-DeadApple-2s-v0',
    entry_point='gym_snake.envs:Snake_16x16_DeadApple_2s'
)
register(
    id='Snake-Hex-16x16-DeadApple-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16_DeadApple_2s'
)
register(
    id='Snake-4x4-2s-v0',
    entry_point='gym_snake.envs:Snake_4x4_2s'
)
register(
    id='Snake-Hex-4x4-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4_2s'
)
register(
    id='Snake-8x8-2s-v0',
    entry_point='gym_snake.envs:Snake_8x8_2s'
)
register(
    id='Snake-Hex-8x8-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8_2s'
)
register(
    id='Snake-16x16-2s-v0',
    entry_point='gym_snake.envs:Snake_16x16_2s'
)
register(
    id='Snake-Hex-16x16-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16_2s'
)
register(
    id='Snake-4x4-Expand-2s-v0',
    entry_point='gym_snake.envs:Snake_4x4_Expand_2s'
)
register(
    id='Snake-Hex-4x4-Expand-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4_Expand_2s'
)
register(
    id='Snake-8x8-Expand-2s-v0',
    entry_point='gym_snake.envs:Snake_8x8_Expand_2s'
)
register(
    id='Snake-Hex-8x8-Expand-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8_Expand_2s'
)
register(
    id='Snake-16x16-Expand-2s-v0',
    entry_point='gym_snake.envs:Snake_16x16_Expand_2s'
)
register(
    id='Snake-Hex-16x16-Expand-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16_Expand_2s'
)
register(
    id='Snake-4x4-4a-2s-v0',
    entry_point='gym_snake.envs:Snake_4x4_4a_2s'
)
register(
    id='Snake-Hex-4x4-4a-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4_4a_2s'
)
register(
    id='Snake-8x8-4a-2s-v0',
    entry_point='gym_snake.envs:Snake_8x8_4a_2s'
)
register(
    id='Snake-Hex-8x8-4a-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8_4a_2s'
)
register(
    id='Snake-16x16-4a-2s-v0',
    entry_point='gym_snake.envs:Snake_16x16_4a_2s'
)
register(
    id='Snake-Hex-16x16-4a-2s-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16_4a_2s'
)
register(
    id='Snake-4x4-DeadApple-3s-v0',
    entry_point='gym_snake.envs:Snake_4x4_DeadApple_3s'
)
register(
    id='Snake-Hex-4x4-DeadApple-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4_DeadApple_3s'
)
register(
    id='Snake-8x8-DeadApple-3s-v0',
    entry_point='gym_snake.envs:Snake_8x8_DeadApple_3s'
)
register(
    id='Snake-Hex-8x8-DeadApple-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8_DeadApple_3s'
)
register(
    id='Snake-16x16-DeadApple-3s-v0',
    entry_point='gym_snake.envs:Snake_16x16_DeadApple_3s'
)
register(
    id='Snake-Hex-16x16-DeadApple-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16_DeadApple_3s'
)
register(
    id='Snake-4x4-3s-v0',
    entry_point='gym_snake.envs:Snake_4x4_3s'
)
register(
    id='Snake-Hex-4x4-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4_3s'
)
register(
    id='Snake-8x8-3s-v0',
    entry_point='gym_snake.envs:Snake_8x8_3s'
)
register(
    id='Snake-Hex-8x8-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8_3s'
)
register(
    id='Snake-16x16-3s-v0',
    entry_point='gym_snake.envs:Snake_16x16_3s'
)
register(
    id='Snake-Hex-16x16-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16_3s'
)
register(
    id='Snake-4x4-Expand-3s-v0',
    entry_point='gym_snake.envs:Snake_4x4_Expand_3s'
)
register(
    id='Snake-Hex-4x4-Expand-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4_Expand_3s'
)
register(
    id='Snake-8x8-Expand-3s-v0',
    entry_point='gym_snake.envs:Snake_8x8_Expand_3s'
)
register(
    id='Snake-Hex-8x8-Expand-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8_Expand_3s'
)
register(
    id='Snake-16x16-Expand-3s-v0',
    entry_point='gym_snake.envs:Snake_16x16_Expand_3s'
)
register(
    id='Snake-Hex-16x16-Expand-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16_Expand_3s'
)
register(
    id='Snake-4x4-4a-3s-v0',
    entry_point='gym_snake.envs:Snake_4x4_4a_3s'
)
register(
    id='Snake-Hex-4x4-4a-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_4x4_4a_3s'
)
register(
    id='Snake-8x8-4a-3s-v0',
    entry_point='gym_snake.envs:Snake_8x8_4a_3s'
)
register(
    id='Snake-Hex-8x8-4a-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_8x8_4a_3s'
)
register(
    id='Snake-16x16-4a-3s-v0',
    entry_point='gym_snake.envs:Snake_16x16_4a_3s'
)
register(
    id='Snake-Hex-16x16-4a-3s-v0',
    entry_point='gym_snake.envs:Snake_Hex_16x16_4a_3s'
)