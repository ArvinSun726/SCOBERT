import os

cur_dir = os.getcwd()


BertBilstmConfig = {
    'max_seq_len': 600, # 128
    'num_classes': 1,
    'word_dict': {'PUSH5' : 0, 'SLOAD' : 1, 'PUSH21': 2, 'SHA3': 3, 'ADDRESS': 4, 'MSTORE8': 5, 'CALLDATASIZE': 6, 'POP': 7, 'PUSH29': 8, 'JUMP': 9, 'PUSH23': 10, 'SUB': 11, 'EXTCODECOPY': 12, 'DUP13': 13, 'SHL': 14, 'RETURNDATACOPY': 15, 'PUSH22': 16, 'GAS': 17, 'DUP16': 18, 'PUSH8': 19, 'PUSH27': 20, 'EXTCODEHASH': 21, 'DUP2': 22, 'DUP14': 23, 'LOG2': 24,
                  'DUP1': 25, 'PUSH28': 26, 'GETPC': 27, 'MSTORE': 28, 'CALLVALUE': 29, 'SWAP1': 30, 'PUSH32': 31, 'BYTE': 32, 'EQ': 33, 'NOT': 34, 'ADD': 35, 'LOG4': 36,
                  'GASLIMIT': 37, 'PUSH19': 38, 'SDIV': 39, 'DUP5': 40, 'SELFDESTRUCT': 41, 'SWAP11': 42, 'PUSH9': 43, 'DIFFICULTY': 44, 'DUP15': 45, 'END': 46, 'SWAP9': 47,
                  'AND': 48, 'PUSH30': 49, 'CALLCODE': 50, 'PUSH6': 51, 'DUP10': 52, 'MLOAD': 53, 'SWAP10': 54, 'SWAP8': 55, 'CREATE': 56, 'SWAP4': 57, 'COINBASE': 58,
                  'PUSH14': 59, 'SWAP7': 60, 'DUP12': 61, 'BALANCE': 62, 'CALL': 63, 'REVERT': 64, 'CALLDATACOPY': 65, 'PUSH18': 66, 'LOG1': 67, 'SGT': 68, 'SWAP13': 69,
                  'PUSH12': 70, 'PUSH10': 71, 'DUP9': 72, 'SWAP16': 73, 'RETURNDATASIZE': 74, 'EXP': 75, 'STATICCALL': 76, 'MOD': 77, 'TIMESTAMP': 78, 'PUSH1': 79,
                  'ORIGIN': 80, 'SWAP15': 81, 'MUL': 82, 'PUSH15': 83, 'SLT': 84, 'CALLDATALOAD': 85, 'PUSH31': 86, 'PUSH16': 87, 'JUMPI': 88, 'BLOCKHASH': 89,
                  'EXTCODESIZE': 90, 'SWAP5': 91, 'PUSH26': 92, 'CODECOPY': 93, 'GASPRICE': 94, 'PUSH17': 95, 'SMOD': 96, 'CREATE2': 97, 'NUMBER': 98, 'SWAP3': 99,
                  'DUP7': 100, 'PUSH20': 101, 'DUP11': 102, 'PUSH24': 103, 'INVALID': 104, 'SWAP14': 105, 'OR': 106, 'PUSH11': 107, 'GT': 108, 'DUP8': 109,
                  'CODESIZE': 110, 'SWAP2': 111, 'STOP': 112, 'DUP3': 113, 'SWAP6': 114, 'PUSH25': 115, 'LT': 116, 'LOG0': 117, 'DELEGATECALL': 118, 'SHR': 119,
                  'PUSH7': 120, 'MSIZE': 121, 'PUSH4': 122, 'DUP4': 123, 'MULMOD': 124, 'RETURN': 125, 'DUP6': 126, 'LOG3': 127, 'PUSH2': 128, 'SWAP12': 129,
                  'CALLER': 130, 'SIGNEXTEND': 131, 'ISZERO': 132, 'SSTORE': 133, 'XOR': 134, 'PUSH13': 135, 'PUSH3': 136, 'DIV': 137, 'ADDMOD': 138, 'JUMPDEST': 139},
    'index_dict':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139],

    'train_sample_path':  os.path.join(cur_dir, "data/train36000.json"),
    'train_sample_path_MaD': os.path.join(cur_dir, "data/MaD_train_test_4000.json"),

    'Ree_train': os.path.join(cur_dir, "data/Ree_train.json"),
    'Ree_test': os.path.join(cur_dir, "data/Ree_test.json"),

    'train_sample_path02': os.path.join(cur_dir, "data/train000.json"),
    'test_sample_path02': os.path.join(cur_dir, "data/test000.json"),
    'vocab_list': ['PUSH5', 'SLOAD', 'PUSH21', 'SHA3', 'ADDRESS', 'MSTORE8', 'CALLDATASIZE', 'POP', 'PUSH29', 'JUMP', 'PUSH23', 'SUB', 'EXTCODECOPY', 'DUP13', 'SHL', 'RETURNDATACOPY', 'PUSH22', 'GAS', 'DUP16', 'PUSH8', 'PUSH27', 'EXTCODEHASH', 'DUP2', 'DUP14', 'LOG2', 'DUP1', 'PUSH28', 'GETPC', 'MSTORE', 'CALLVALUE', 'SWAP1', 'PUSH32', 'BYTE', 'EQ', 'NOT', 'ADD', 'LOG4', 'GASLIMIT', 'PUSH19', 'SDIV', 'DUP5', 'SELFDESTRUCT', 'SWAP11', 'PUSH9', 'DIFFICULTY', 'DUP15', 'END', 'SWAP9', 'AND', 'PUSH30', 'CALLCODE', 'PUSH6', 'DUP10', 'MLOAD', 'SWAP10', 'SWAP8', 'CREATE', 'SWAP4', 'COINBASE', 'PUSH14', 'SWAP7', 'DUP12', 'BALANCE', 'CALL', 'REVERT', 'CALLDATACOPY', 'PUSH18', 'LOG1', 'SGT', 'SWAP13', 'PUSH12', 'PUSH10', 'DUP9', 'SWAP16', 'RETURNDATASIZE', 'EXP', 'STATICCALL', 'MOD', 'TIMESTAMP', 'PUSH1', 'ORIGIN', 'SWAP15', 'MUL', 'PUSH15', 'SLT', 'CALLDATALOAD', 'PUSH31', 'PUSH16', 'JUMPI', 'BLOCKHASH', 'EXTCODESIZE', 'SWAP5', 'PUSH26', 'CODECOPY', 'GASPRICE', 'PUSH17', 'SMOD', 'CREATE2', 'NUMBER', 'SWAP3', 'DUP7', 'PUSH20', 'DUP11', 'PUSH24', 'INVALID', 'SWAP14', 'OR', 'PUSH11', 'GT', 'DUP8', 'CODESIZE', 'SWAP2', 'STOP', 'DUP3', 'SWAP6', 'PUSH25', 'LT', 'LOG0', 'DELEGATECALL', 'SHR', 'PUSH7', 'MSIZE', 'PUSH4', 'DUP4', 'MULMOD', 'RETURN', 'DUP6', 'LOG3', 'PUSH2', 'SWAP12', 'CALLER', 'SIGNEXTEND', 'ISZERO', 'SSTORE', 'XOR', 'PUSH13', 'PUSH3', 'DIV', 'ADDMOD', 'JUMPDEST']







}