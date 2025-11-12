def one_test_depth(inplace, D, H, W, kernel_size, stride, padding, numthreads):
    we = np.zeros(D*kernel_size*kernel_size)
    for i in range(0, len(we)):
        die_roll = np.random.uniform()
        if die_roll < 1/3:
            we[i] = -1
        if die_roll > 2/3:
            we[i] = 1
    we = np.reshape(we, (D, kernel_size, kernel_size))
    wec = np.zeros(D*kernel_size*kernel_size)
    cwec = 0
    for h in range(0, kernel_size):
        for w in range(0, kernel_size):
            for d in range(0, D):
                wec[cwec] = we[d, h, w]
                cwec += 1
    with open('wd.txt', 'w') as f:
        for i in range(0, len(wec)):
            f.write(f'{int(wec[i])}\n')
    we = np.reshape(we, (D, 1, kernel_size, kernel_size))
    wet = torch.from_numpy(we)
    wet = torch.tensor(wet, dtype = torch.float32)
    inp = torch.randint(-256, 256, (D, H, W))
    inp_padded = torch.zeros((D, H+2*padding, W+2*padding), dtype=torch.int32)
    inp_padded[:, padding:padding+H, padding:padding+W] = inp
    inp = torch.tensor(inp, dtype=torch.float32)
    thingToWrite = inp_padded
    if inplace:
        thingToWrite = inp
    with open('inpd.txt', 'w') as f:
        for item in thingToWrite.flatten():
            f.write(f'{int(item.item())}\n') 
    with open('dparams.txt', 'w') as f:
        f.write(f'{int(not inplace)} {numthreads} {1} {D} {H} {W} {kernel_size} {stride} {padding}')
    outpr = torch.nn.functional.conv2d(inp, wet, None, stride, padding, 1, D)
    #print(outpr)
    if D > 1:
        outpr = torch.squeeze(outpr, 0)
    outp = torch.zeros(outpr.shape[0]*outpr.shape[1]*outpr.shape[2])
    coutp = 0
    for i in range(0, outpr.shape[1]):
        for j in range(0, outpr.shape[2]):
            for h in range(0, outpr.shape[0]):
                outp[coutp] = outpr[h, i, j]
                coutp += 1
    !time ./depth_code
    outp_from_code = np.loadtxt("output.txt")
    outp_from_code = torch.from_numpy(outp_from_code)
    print("len real output = ", len(outp))
    print("len c output = ", len(outp_from_code))
    diff = torch.abs(outp - outp_from_code)
    for i in range(0, len(diff)):
        if diff[i].item() > 0.01:
            pass
            print(f"hm, {i}, {outp[i].item()}, {outp_from_code[i].item()}")
            #break