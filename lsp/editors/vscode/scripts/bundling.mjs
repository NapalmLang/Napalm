import {openSync, readSync, closeSync, statSync} from 'node:fs';

export const supportedTargets = new Set([
    'win32-x64', 'win32-arm64',
    'linux-x64', 'linux-arm64',
    'darwin-x64', 'darwin-arm64',
]);

export function detectTarget(platform = process.platform, arch = process.arch) {
    const normalized = `${platform}-${arch}`;
    if (!supportedTargets.has(normalized)) {
        throw new Error(`Unsupported host ${normalized}; specify a supported --target and executable.`);
    }
    return normalized;
}

function readPrefix(file, bytes = 4096) {
    const fd = openSync(file, 'r');
    try {
        const prefix = Buffer.alloc(bytes);
        const read = readSync(fd, prefix, 0, prefix.length, 0);
        return prefix.subarray(0, read);
    } finally {
        closeSync(fd);
    }
}

export function identifyExecutable(file) {
    const bytes = readPrefix(file);
    if (bytes.length >= 64 && bytes.readUInt16LE(0) === 0x5A4D) {
        const offset = bytes.readUInt32LE(0x3c);
        if (offset + 6 > bytes.length || bytes.readUInt32LE(offset) !== 0x00004550) {
            throw new Error('Invalid Windows PE executable header');
        }
        const machine = bytes.readUInt16LE(offset + 4);
        const arch = machine === 0x8664 ? 'x64' : machine === 0xaa64 ? 'arm64' : null;
        if (!arch) throw new Error(`Unsupported PE architecture: 0x${machine.toString(16)}`);
        return `win32-${arch}`;
    }
    if (bytes.length >= 20 && bytes.subarray(0, 4).equals(Buffer.from([0x7f, 0x45, 0x4c, 0x46]))) {
        if (bytes[4] !== 2 || bytes[5] !== 1) {
            throw new Error('Expected a little-endian 64-bit ELF executable');
        }
        const machine = bytes.readUInt16LE(18);
        const arch = machine === 0x3e ? 'x64' : machine === 0xb7 ? 'arm64' : null;
        if (!arch) throw new Error(`Unsupported ELF architecture: ${machine}`);
        return `linux-${arch}`;
    }
    if (bytes.length >= 8 && bytes.subarray(0, 4).equals(Buffer.from([0xcf, 0xfa, 0xed, 0xfe]))) {
        const cpuType = bytes.readUInt32LE(4);
        const arch = cpuType === 0x01000007 ? 'x64' : cpuType === 0x0100000c ? 'arm64' : null;
        if (!arch) throw new Error(`Unsupported Mach-O CPU type: 0x${cpuType.toString(16)}`);
        return `darwin-${arch}`;
    }
    throw new Error('The supplied file is not a supported executable (PE64, ELF64 or Mach-O 64)');
}

export function validateExecutable(file, target) {
    if (!supportedTargets.has(target)) throw new Error(`Unsupported VS Code target: ${target}`);
    const size = statSync(file).size;
    if (size < 512) throw new Error(`The server binary is unexpectedly small: ${size} bytes`);
    const actual = identifyExecutable(file);
    if (actual !== target) {
        throw new Error(`Binary is for ${actual}, but VSIX target is ${target}`);
    }
    return size;
}

// Read only the ZIP central directory, avoiding loading a large embedded LLVM
// executable into memory just to validate the final package.
export function getVsixEntry(vsix, expectedName) {
    const fd = openSync(vsix, 'r');
    try {
        const length = statSync(vsix).size;
        const tailLength = Math.min(length, 65557);
        const tail = Buffer.alloc(tailLength);
        readSync(fd, tail, 0, tailLength, length - tailLength);
        let eocd = -1;
        for (let p = tail.length - 22; p >= 0; p--) {
            if (tail.readUInt32LE(p) === 0x06054b50 && p + 22 + tail.readUInt16LE(p + 20) === tail.length) {
                eocd = p;
                break;
            }
        }
        if (eocd < 0) throw new Error('Invalid VSIX: ZIP end-of-central-directory record not found');
        const entryCount = tail.readUInt16LE(eocd + 10);
        const directoryLength = tail.readUInt32LE(eocd + 12);
        const directoryOffset = tail.readUInt32LE(eocd + 16);
        if (entryCount === 0xffff || directoryLength === 0xffffffff || directoryOffset === 0xffffffff) {
            throw new Error('ZIP64 VSIX is not supported by this packaging verifier');
        }
        if (directoryOffset + directoryLength > length) {
            throw new Error('Invalid VSIX central directory bounds');
        }
        const directory = Buffer.alloc(directoryLength);
        readSync(fd, directory, 0, directoryLength, directoryOffset);
        let offset = 0;
        for (let i = 0; i < entryCount; i++) {
            if (offset + 46 > directory.length || directory.readUInt32LE(offset) !== 0x02014b50) {
                throw new Error('Invalid ZIP central directory entry');
            }
            const nameLength = directory.readUInt16LE(offset + 28);
            const extraLength = directory.readUInt16LE(offset + 30);
            const commentLength = directory.readUInt16LE(offset + 32);
            const end = offset + 46 + nameLength + extraLength + commentLength;
            if (end > directory.length) throw new Error('Truncated ZIP central directory entry');
            const name = directory.toString('utf8', offset + 46, offset + 46 + nameLength);
            if (name === expectedName) {
                return {name, uncompressedSize: directory.readUInt32LE(offset + 24)};
            }
            offset = end;
        }
        return null;
    } finally {
        closeSync(fd);
    }
}
