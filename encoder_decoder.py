import chess
import chess.pgn
import torch 
import numpy as np

def encode(board):
    encoded = torch.zeros((22,8,8), dtype = torch.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    
    #12 channels -> 6 for white pieces[0-5] + 6 for black pieces[6-11]
    for square, piece in board.piece_map().items():
        offset = 6 if piece.color == chess.BLACK else 0
        piece_channel = piece_map[piece.piece_type] + offset
        rank, file = divmod(square, 8)
        encoded[piece_channel, rank, file] = 1
    
    #4 channels for castling rights[12-15]
    encoded[12,:,:] = board.has_kingside_castling_rights(chess.WHITE)
    encoded[13,:,:] = board.has_queenside_castling_rights(chess.WHITE)
    encoded[14,:,:] = board.has_kingside_castling_rights(chess.BLACK)
    encoded[15,:,:] = board.has_queenside_castling_rights(chess.BLACK)

    #1 channel en passant [16]
    if board.ep_square:
        ep_rank, ep_file = divmod(board.ep_square,8)
        encoded [16,ep_rank,ep_file] = 1

    #1 channel player turn -> 1 for white 0 for black
    encoded[17,:,:] = board.turn

    # Halfmove clock and fullmove number (normalized) -> 2 channels[18,19] 
    encoded[18, :, :] = board.halfmove_clock / 50.0
    encoded[19, :, :] = board.fullmove_number / 100.0

    # Threefold repetition rule -> 2 channels[20,21]
    encoded[20, :, :] = board.is_repetition(1)
    encoded[21, :, :] = board.is_repetition(2)

    return encoded      #-> 12+4+1+1+2+2 = 22 channels


def show_encoding(encoded):
    C,R,F = encoded.shape
    print(C,R,F)
    for i in range(C):
        print('\n'+ '='*40 + '\n')
        print(f'channel no: {i}')
        print(encoded[i,:,:])



def encode_action(piece, initial_pos, final_pos, underpromote = None):
    i, j = initial_pos
    x, y = final_pos
    dx, dy = x - i, y - j
    idx = None
    
    if piece in ['P', 'R', 'B', 'Q', 'K', 'p', 'r', 'b', 'q', 'k'] and underpromote is None:
        if dx != 0 and dy == 0: #idx=[0,1,2,3,4,5,6]black  idx=[7,8,9,10,11,12,13]white   North-South
            idx = 7 + dx if dx < 0 else 6 + dx
        elif dx == 0 and dy != 0:  #idx=[14,15,16,17,18,19,20]West->East idx=[21,22,23,24,25,26,27]East->West  
            idx = 21 + dy if dy < 0 else 20 + dy
        elif dx == dy:  # idx[28,29,30,31,32,33,34]SW->NE   idx[35,36,37,38,39,40,41]SW<-NE
            idx = 35 + dx if dx < 0 else 34 + dx
        elif dx == -dy:  # idx[42,43,44,45,46,47,48]SE->NW   idx[49,50,51,52,53,54,55]SE<-NW
            idx = 49 + dx if dx < 0 else 48 + dx


    elif piece in ["n", "N"]:  # Knight moves idx=[56,57,58,59,60,61,62,63]
        knight_moves = {(2, -1): 56, (2, 1): 57, (1, -2): 58, (-1, -2): 59,
                        (-2, 1): 60, (-2, -1): 61, (-1, 2): 62, (1, 2): 63}
        idx = knight_moves.get((dx, dy))
    
    elif piece in ["p", "P"] and (x == 0 or x == 7) and underpromote is not None:  # Underpromotions idx = [64, 65, 66, 67, 68, 69, 70, 71, 72, 73,74,75]
        # print('this is triggered')
        underpromotion_map = {
            (4, 0): 64,   # ROOK,  straight ahead
            (2, 0): 65,   # KNIGHT, straight ahead
            (3, 0): 66,   # BISHOP, straight ahead
            (5, 0): 67,   # QUEEN,  straight ahead
            (4, -1): 68,  # ROOK,   capture left
            (2, -1): 69,  # KNIGHT, capture left
            (3, -1): 70,  # BISHOP, capture left
            (5, -1): 71,  # QUEEN,  capture left
            (4, 1): 72,   # ROOK,   capture right
            (2, 1): 73,   # KNIGHT, capture right
            (3, 1): 74,   # BISHOP, capture right
            (5, 1): 75    # QUEEN,  capture right
        }
        
        idx = underpromotion_map.get((underpromote, dy))

    if idx is not None: 
        return idx, i, j
    else:
        return ValueError(f"Some error in encode function ")


def decode_action(encoded):
    encoded = encoded.view(76,8,8)
    C,R,F = torch.where(encoded > 0)
    C, R, F = C.tolist(), R.tolist(), F.tolist()
    probabilities = encoded[C, R, F].tolist()
    prom, i_pos, f_pos = [],[],[]

    for c,r,f in zip(C,R,F):
        initial_pos = (r,f)
        final_pos = None
        promoted = None

        if 0 <= c <= 13:
            dx = c - 7 if c < 7 else c - 6
            final_pos = (r + dx, f)
        elif 14 <= c <= 27:
            dy = c - 21 if c < 21 else c - 20
            final_pos = (r, f + dy)
        elif 28 <= c <= 41:
            dy = c - 35 if c < 35 else c - 34
            final_pos = (r + dy, f + dy)
        elif 42 <= c <= 55:
            dx = c - 49 if c < 49 else c - 48
            dy = -dx
            final_pos = (r + dx, f + dy)
        elif 56 <= c <= 63:  
            knight_moves = {
                56: (r+2, f-1), 57: (r+2, f+1), 58: (r+1, f-2), 59: (r-1, f-2),
                60: (r-2, f+1), 61: (r-2, f-1), 62: (r-1, f+2), 63: (r+1, f+2)
            }
            final_pos = knight_moves[c]
        elif 64<=c<=75:
            underpromote_map = {
                64: ('rook', 0), 65: ('knight', 0), 66: ('bishop', 0), 67: ('queen', 0),
                68: ('rook', -1), 69: ('knight', -1), 70: ('bishop', -1), 71: ('queen', -1),
                72: ('rook', 1), 73: ('knight', 1), 74: ('bishop', 1), 75: ('queen', 1),
            }
            promoted, dy = underpromote_map[c]
            final_pos = (r,f+dy)
        
        i_pos.append(initial_pos)
        f_pos.append(final_pos)
        prom.append(promoted)

    return i_pos, f_pos, prom, probabilities



def encode_legal_moves(board, log):
    legal_mask = torch.zeros((76, 8, 8), dtype=torch.uint8)
    for move in board.legal_moves:
        i, j = divmod(move.from_square, 8)
        x, y = divmod(move.to_square, 8)
        piece = board.piece_at(move.from_square)
        
        if piece is not None:  
            idx, x, y = encode_action(piece.symbol(), (i, j), (x, y), move.promotion)
            if log: print(f"Encoding move: {piece} {move} -> idx {idx}, position ({x}, {y})")            
            legal_mask[idx, x, y] = 1
            if log: print(legal_mask[idx,:,:])
    return legal_mask.view(76*8*8)


def decode_from_index(index: int) -> chess.Move:
    c, board = divmod(index, 64)
    r,f = divmod(board,8)
    initial_pos = (r,f)
    final_pos = None
    promoted = None

    if 0 <= c <= 13:
        dx = c - 7 if c < 7 else c - 6
        final_pos = (r + dx, f)
    elif 14 <= c <= 27:
        dy = c - 21 if c < 21 else c - 20
        final_pos = (r, f + dy)
    elif 28 <= c <= 41:
        dy = c - 35 if c < 35 else c - 34
        final_pos = (r + dy, f + dy)
    elif 42 <= c <= 55:
        dx = c - 49 if c < 49 else c - 48
        dy = -dx
        final_pos = (r + dx, f + dy)
    elif 56 <= c <= 63:  
        knight_moves = {
            56: (r+2, f-1), 57: (r+2, f+1), 58: (r+1, f-2), 59: (r-1, f-2),
            60: (r-2, f+1), 61: (r-2, f-1), 62: (r-1, f+2), 63: (r+1, f+2)
        }
        final_pos = knight_moves[c]
    elif 64<=c<=75:
        underpromote_map = {
            64: (4, 0),   # ROOK,  straight ahead
            65: (2, 0),   # KNIGHT, straight ahead
            66: (3, 0),   # BISHOP, straight ahead
            67: (5, 0),   # QUEEN,  straight ahead
            68: (4, -1),  # ROOK,   capture left
            69: (2, -1),  # KNIGHT, capture left
            70: (3, -1),  # BISHOP, capture left
            71: (5, -1),  # QUEEN,  capture left
            72: (4, 1),   # ROOK,   capture right
            73: (2, 1),   # KNIGHT, capture right
            74: (3, 1),   # BISHOP, capture right
            75: (5, 1),   # QUEEN,  capture right
        }
        dx = -1 if initial_pos[0]==1 else 1
        promoted, dy = underpromote_map[c]
        final_pos = (r+dx,f+dy)

    # print(f'initial_position: {initial_pos}')
    # print(f'final_position: {final_pos}')
    # print(f'promotes: {promoted}')

    return make_move(initial_pos, final_pos, promoted)



def make_move(initial_pos, final_pos, promoted=None):
    from_square = chess.square(initial_pos[1], initial_pos[0])
    to_square = chess.square(final_pos[1], final_pos[0])
    promotion_piece = promoted
    return chess.Move(from_square, to_square, promotion=promotion_piece)

