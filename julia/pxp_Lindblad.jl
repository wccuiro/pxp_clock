using ITensors, ITensorMPS

function L_op_j(i)
  os_plus = OpSum()
  
  im1 = mod1(i - 1, N)
  ip1 = mod1(i + 1, N)
  
  os_plus += 0.125, "Id", im1, "X", i, "Id", ip1
  os_plus -= 0.125, "Id", im1, "X", i, "Z", ip1
  os_plus += 0.125im, "Id", im1, "Y", i, "Id", ip1
  os_plus -= 0.125im, "Id", im1, "Y", i, "Z", ip1
  os_plus -= 0.125, "Z", im1, "X", i, "Id", ip1
  os_plus += 0.125, "Z", im1, "X", i, "Z", ip1
  os_plus -= 0.125im, "Z", im1, "Y", i, "Id", ip1
  os_plus += 0.125im, "Z", im1, "Y", i, "Z", ip1

  os_minus = OpSum()

  os_minus += 0.125, "Id", im1, "X", i, "Id", ip1
  os_minus -= 0.125, "Id", im1, "X", i, "Z", ip1
  os_minus -= 0.125im, "Id", im1, "Y", i, "Id", ip1
  os_minus += 0.125im, "Id", im1, "Y", i, "Z", ip1
  os_minus -= 0.125, "Z", im1, "X", i, "Id", ip1
  os_minus += 0.125, "Z", im1, "X", i, "Z", ip1
  os_minus += 0.125im, "Z", im1, "Y", i, "Id", ip1
  os_minus -= 0.125im, "Z", im1, "Y", i, "Z", ip1

  return os_plus, os_minus
end

function L_minus_j(i)
  os = OpSum()
  im1 = mod1(i - 1, N)
  ip1 = mod1(i + 1, N)
  os += 0.125, "Id", im1, "X", i, "Id", ip1
  os -= 0.125, "Id", im1, "X", i, "Z", ip1
  os += 0.125im, "Id", im1, "Y", i, "Id", ip1
  os -= 0.125im, "Id", im1, "Y", i, "Z", ip1
  os -= 0.125, "Z", im1, "X", i, "Id", ip1
  os += 0.125, "Z", im1, "X", i, "Z", ip1
  os -= 0.125im, "Z", im1, "Y", i, "Id", ip1
  os += 0.125im, "Z", im1, "Y", i, "Z", ip1
  return os
end


function main()


end

# Run the main function
main()